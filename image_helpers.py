import os
import cv2
import torch
import numpy as np
import dotenv

dotenv.load_dotenv()

from lang_sam import LangSAM
from concurrent.futures import ThreadPoolExecutor
from diffusers import AutoPipelineForInpainting, FluxTransformer2DModel
from diffusers.utils import make_image_grid
from PIL import Image
from copy import deepcopy

from utils.utils import mask_to_rectangle_mask, mask_to_square_mask, mask_to_bbox
from models.omnieraser.pipeline_flux_control_removal import FluxControlRemovalPipeline

#################################################################
# Initializations
#################################################################

os.environ["TOKENIZERS_PARALLELISM"] = "false"
enable_offload = not bool(os.getenv("DISABLE_OFFLOAD"))

def load_gsam():
    global model
    if "model" in globals() and model is not None:
        pass
    else:
        model = LangSAM(sam_type="sam2.1_hiera_large")
        if enable_offload:
            model.sam.model.cpu()
            model.gdino.model.cpu()
        else:
            model.sam.model.cuda()
            model.gdino.model.cuda()

def delete_gsam():
    global model
    if "model" in globals() and model is not None:
        del model
        model = None
        torch.cuda.empty_cache()
        print("Deleted gsam model")
    else:
        print("gsam model not loaded")


def load_omni():
    global pipe
    if "pipe" in globals() and pipe is not None:
        pass
    else:
        # Build pipeline
        transformer = FluxTransformer2DModel.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )

        with torch.no_grad():
            initial_input_channels = transformer.config.in_channels
            new_linear = torch.nn.Linear(
                transformer.x_embedder.in_features * 4,
                transformer.x_embedder.out_features,
                bias=transformer.x_embedder.bias is not None,
                dtype=transformer.dtype,
                device=transformer.device,
            )
            new_linear.weight.zero_()
            new_linear.weight[:, :initial_input_channels].copy_(
                transformer.x_embedder.weight
            )
            if transformer.x_embedder.bias is not None:
                new_linear.bias.copy_(transformer.x_embedder.bias)
            transformer.x_embedder = new_linear
            transformer.register_to_config(in_channels=initial_input_channels * 4)

        pipe = FluxControlRemovalPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        pipe.transformer.to(torch.bfloat16)
        assert pipe.transformer.config.in_channels == initial_input_channels * 4, (
            f"{pipe.transformer.config.in_channels=}"
        )

        pipe.load_lora_weights(
            "theSure/Omnieraser",
            weight_name="pytorch_lora_weights.safetensors",
        )
        if enable_offload:
            pipe.to("cpu")
        else:
            pipe.to("cuda")


def delete_omni():
    global pipe
    if "pipe" in globals() and pipe is not None:
        del pipe
        pipe = None
        torch.cuda.empty_cache()
        print("Deleted omni model")
    else:
        print("omni model not loaded")

def get_inpainted_image_omni(image, mask):
    """
    Inpaints an image using a diffusion pipeline with a control mask.
    This function takes an input image and a corresponding mask, resizes them to 1024x1024,
    and then applies an inpainting process using a pre-configured diffusion model pipeline.
    A fixed prompt ("There is nothing here.") and a constant random seed are used for the inpainting.
    After processing, the inpainted image is resized back to the original image dimensions.
    Parameters:
        image (PIL.Image.Image): The input image to be inpainted.
        mask (PIL.Image.Image): The mask indicating areas to modify during inpainting.
    Returns:
        PIL.Image.Image: The resulting inpainted image resized to the original size.
    Notes:
        - The function uses a global flag 'enable_offload' to determine whether to move the pipeline
          to and from the GPU before and after processing.
        - It is assumed that the diffusion pipeline 'pipe' and the flag 'enable_offload' are defined
          in the global scope.
        - A torch.Generator is created with a fixed seed (42) for reproducibility.
    """

    if enable_offload:
        pipe.to("cuda")

    prompt = "There is nothing here."

    original_size = image.size
    image = image.convert("RGB").resize((1024, 1024), Image.LANCZOS)
    mask = mask.convert("RGB").resize((1024, 1024), Image.LANCZOS)

    generator = torch.Generator(device="cuda").manual_seed(42)

    # Inpaint
    result = pipe(
        prompt=prompt,
        control_image=image,
        control_mask=mask,
        num_inference_steps=28,
        guidance_scale=3.5,
        generator=generator,
        max_sequence_length=512,
        height=image.size[1],
        width=image.size[0],
    ).images[0]

    # Resize back to original size
    result = result.resize(original_size, Image.LANCZOS)

    if enable_offload:
        pipe.to("cpu")
        torch.cuda.empty_cache()

    return result

def get_mask_and_bbox(
    image, mask_type=None, text_prompt="person.", dilate_iterations=4
):
    """
    Get the image and mask of the object in the image that is grounded to the text prompt.

    Args:
        image (PIL.Image): image
        mask_type (str): type of the mask, either "rectangle" or "square", or None for the original mask
        text_prompt (str): text prompt to ground the object

    Returns:
        tuple: a tuple of (image, mask, mask_org), where image is a PIL image object, and mask is a numpy array with values in [0, 255],
        processed using the mask_type, and mask_org is the original mask without any processing.
    """

    # use the batch version of the function
    masks, boxes, confs, masks_org = get_masks_and_bboxes(
        [image], mask_type, text_prompt, dilate_iterations
    )

    return masks[0], boxes[0], confs[0], masks_org[0]


def process_masks(masks, mask_type, dilate_iterations=3):
    """
    Process the masks to make them suitable for inpainting.

    Args:
        masks (list): list of numpy arrays with values in [0, 255]
        mask_type (str): type of the mask, either "rectangle" or "square", or None for the original mask

    Returns:
        list: a list of processed masks
    """

    # dilate the mask to make the inpainting more realistic
    kernel = np.ones((9, 9), np.uint8)
    masks = [cv2.dilate(mask, kernel, iterations=dilate_iterations) for mask in masks]

    # convert the mask into bounding box
    if mask_type == "rectangle":
        masks = [mask_to_rectangle_mask(mask) for mask in masks]
    elif mask_type == "square":
        masks = [mask_to_square_mask(mask) for mask in masks]

    return masks


def get_masks_and_bboxes(
    images, mask_type=None, text_prompt="person.", dilate_iterations=4
):
    """
    Get the images and masks of the objects in the images that are grounded to the text prompt.

    Args:
        images(list of PIL.Image): list of images
        mask_type (str): type of the mask, either "rectangle" or "square", or None for the original mask
        text_prompt (str): text prompt to ground the object

    Returns:
        tuple: a tuple of (images, masks, masks_org), where images is a list of PIL image objects, and masks is a list of numpy arrays with values in [0, 255],
        processed using the mask_type, and masks_org is the original masks without any processing.
    """

    if text_prompt[-1] != ".":
        text_prompt += "."

    if enable_offload:
        model.sam.model.cuda()
        model.gdino.model.cuda()

    captions = [text_prompt] * len(images)

    results = model.predict(images, captions)
    masks = [result["masks"] for result in results]
    boxes = [result["boxes"] for result in results]  # N_humans, 4
    confs = [result["scores"] for result in results]  # N_humans

    for idx, mask in enumerate(masks):
        if isinstance(mask, list) and len(mask) == 0:
            w, h = images[idx].size
            masks[idx] = np.zeros((1, h, w), dtype=bool)

    masks = [mask.astype(np.uint8) * 255 for mask in masks]
    masks = [
        list(mask) for mask in masks
    ]  # convert (N_humans, H, W) to list of (H, W) for each image

    # make a copy of the masks before processing
    masks_org = deepcopy(masks)

    # dilate the mask to make the inpainting more realistic
    # masks = process_masks(masks, mask_type, dilate_iterations)
    masks = [
        process_masks(masks_image, mask_type, dilate_iterations)
        for masks_image in masks
    ]

    if enable_offload:
        model.sam.model.cpu()
        model.gdino.model.cpu()

    return masks, boxes, confs, masks_org


def fill_mask_with_background_color(image, mask):
    np_image = np.array(image)
    bool_mask = mask > 127

    # Identify mask borders (background pixels that neighbor the mask)
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(bool_mask.astype(np.uint8), kernel, iterations=1).astype(
        bool
    )
    border_bg = dilated_mask & (~bool_mask)

    # Calculate fill color based on border BG pixels
    bg_pixels = np_image[border_bg]
    if bg_pixels.size == 0:
        fill_color = (255, 255, 255)
    else:
        fill_color = np.mean(bg_pixels, axis=0).astype(np.uint8)

    np_image[bool_mask] = fill_color
    return Image.fromarray(np_image)


def fill_masks_with_background_color(images, masks):
    """
    Parallely fills the masks with the background color of the images.

    Args:
    images (list): list of PIL images
    masks (list): list of numpy arrays with values in [0, 255]

    Returns:
    list: a list of PIL images, where the i-th image is inpainted using the i-th mask
    """

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(fill_mask_with_background_color, images, masks))

    return results
