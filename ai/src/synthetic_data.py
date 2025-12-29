import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image
import os
import logging
from typing import Optional, List, Tuple
import numpy as np
from torchvision import transforms
import cv2
import glob
import random
from scipy.ndimage import gaussian_filter, map_coordinates

logger = logging.getLogger(__name__)

class UltrasoundGenerator:
    """Generator for synthetic ultrasound images using Stable Diffusion"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to prevent multiple model loads"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: Optional[str] = None,
        template_dir: Optional[str] = None
    ):
        """Initialize the ultrasound image generator
        
        Args:
            model_id: Hugging Face model ID for Stable Diffusion
            device: Device to run generation on (cuda/cpu)
            template_dir: Directory containing real ultrasound templates
        """
        # Only initialize once
        if self._initialized:
            return
            
        self.device = device if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Set number of CPU threads
        if self.device == "cpu":
            torch.set_num_threads(8)  # Adjust based on available CPU cores
            
        # Initialize the pipelines
        self.txt2img = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            safety_checker=None  # Disable safety checker for medical images
        ).to(self.device)
        
        self.img2img = StableDiffusionImg2ImgPipeline(
            vae=self.txt2img.vae,
            text_encoder=self.txt2img.text_encoder,
            tokenizer=self.txt2img.tokenizer,
            unet=self.txt2img.unet,
            scheduler=self.txt2img.scheduler,
            safety_checker=None,  # Disable safety checker for medical images
            feature_extractor=self.txt2img.feature_extractor
        ).to(self.device)
        
        # Load template images if available
        self.template_dir = template_dir
        self.templates = []
        if template_dir:
            for img_path in glob.glob(os.path.join(template_dir, "*.jpg")) + \
                            glob.glob(os.path.join(template_dir, "*.png")):
                try:
                    img = Image.open(img_path).convert('RGB')
                    # Resize template images to 256x256 for faster processing
                    img = img.resize((256, 256), Image.Resampling.LANCZOS)
                    self.templates.append(img)
                except Exception as e:
                    logger.warning(f"Failed to load template image {img_path}: {str(e)}")
        
        if not self.templates and template_dir:
            raise ValueError("No valid template images found in directory")
            
        # Configure pipelines
        self.txt2img.set_progress_bar_config(disable=True)
        self.img2img.set_progress_bar_config(disable=True)
        if self.device == "cpu":
            self.txt2img.enable_attention_slicing()
            self.img2img.enable_attention_slicing()
            
        self._initialized = True
    
    def _load_templates(self) -> List[Image.Image]:
        """Load real ultrasound images as templates"""
        templates = []
        if not os.path.exists(self.template_dir):
            return templates
            
        for img_path in glob.glob(os.path.join(self.template_dir, "*.jpg")) + \
                        glob.glob(os.path.join(self.template_dir, "*.png")):
            try:
                img = Image.open(img_path).convert('RGB')
                templates.append(img)
            except Exception as e:
                logger.warning(f"Failed to load template image {img_path}: {str(e)}")
        
        if not templates:
            raise ValueError("No valid template images found in directory")
            
        return templates
    
    def _preprocess_template(self, image: Image.Image) -> Image.Image:
        """Preprocess template image to enhance cervical features"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            
        # Apply CLAHE to enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply histogram equalization to further improve contrast
        hist_eq = cv2.equalizeHist(enhanced)
        balanced = cv2.addWeighted(enhanced, 0.7, hist_eq, 0.3, 0)
        
        # Denoise while preserving edges
        denoised = cv2.fastNlMeansDenoising(balanced, None, 10, 7, 21)
        
        # Apply unsharp masking to enhance edges
        gaussian = cv2.GaussianBlur(denoised, (0, 0), 3.0)
        unsharp_mask = cv2.addWeighted(denoised, 2.0, gaussian, -1.0, 0)
        
        # Enhance edges to make cervical boundaries more visible
        edges = cv2.Canny(unsharp_mask, 30, 150)
        kernel = np.ones((2,2), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Combine enhanced image with edges
        enhanced_with_edges = cv2.addWeighted(unsharp_mask, 0.8, dilated_edges, 0.2, 0)
        
        # Convert back to RGB for stable diffusion input
        enhanced_rgb = cv2.cvtColor(enhanced_with_edges, cv2.COLOR_GRAY2RGB)
        
        return Image.fromarray(enhanced_rgb)
        
    def _get_prompt_for_dilation(self, dilation_cm: float) -> str:
        """Generate appropriate prompt based on dilation measurement"""
        base_prompt = (
            "transvaginal ultrasound scan of cervix, medical grayscale imaging, "
            "professional diagnostic quality, hyperechoic cervical tissue"
        )
        
        if dilation_cm < 2:
            details = (
                "closed cervical os, thick cervical walls showing hyperechoic density, "
                "minimal cervical effacement, narrow cervical canal, strong shadowing"
            )
        elif dilation_cm < 4:
            details = (
                "partially open cervical os, moderately thick cervical walls with effacement, "
                "cervical canal widening, early fetal head engagement, mixed echogenicity"
            )
        elif dilation_cm < 7:
            details = (
                "widely open cervical os, thinning cervical walls with effacement, "
                "prominent hypoechoic canal, clear fetal head engagement, decreased density"
            )
        else:
            details = (
                "fully dilated cervical os, maximally thinned cervical walls, "
                "very wide hypoechoic opening, complete fetal engagement, minimal shadowing"
            )
        
        return f"{base_prompt}, {details}"

    def _add_ultrasound_characteristics(self, image: np.ndarray, dilation_cm: float) -> np.ndarray:
        """Add realistic ultrasound characteristics"""
        # Convert to float for processing
        img = image.astype(np.float32)
        height, width = img.shape
        
        # 1. Add depth-dependent attenuation
        depth_factor = np.linspace(1.0, 0.6, height)  # Stronger depth attenuation
        depth_factor = depth_factor.reshape(-1, 1)
        img *= depth_factor
        
        # 2. Add curved probe effect (more pronounced)
        x = np.linspace(-1, 1, width)
        curve = 1 - 0.35 * x**2  # Increased curve effect
        img *= curve.reshape(1, -1)
        
        # 3. Add anatomical shadowing
        center_x = width // 2
        center_y = int(height * 0.6)
        
        # Create shadow mask based on dilation
        shadow_mask = np.zeros_like(img)
        shadow_strength = max(0.2, 0.5 - (dilation_cm / 20))  # Decreases with dilation
        shadow_size = int(60 - (dilation_cm * 3))  # Shadow size decreases with dilation
        
        # Posterior acoustic shadowing
        y, x = np.ogrid[:height, :width]
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        shadow_mask[dist < shadow_size] = shadow_strength
        
        # Add lateral shadows for cervical walls
        wall_width = int(35 * (1 - dilation_cm/10))  # Walls thin as dilation increases
        wall_shadow = np.zeros_like(img)
        wall_shadow[:, center_x-wall_width:center_x] = 0.3
        wall_shadow[:, center_x+wall_width:center_x+wall_width*2] = 0.3
        
        # Blur shadows
        shadow_mask = cv2.GaussianBlur(shadow_mask, (15, 15), 5)
        wall_shadow = cv2.GaussianBlur(wall_shadow, (11, 11), 3)
        
        # Apply shadows
        img *= (1 - shadow_mask)
        img *= (1 - wall_shadow)
        
        # 4. Add depth-dependent speckle noise
        noise_intensity = np.linspace(0.02, 0.08, height).reshape(-1, 1)  # Increased noise with depth
        speckle = np.random.normal(0, noise_intensity, img.shape)
        img += speckle
        
        # 5. Add tissue interface enhancement
        edges = cv2.Canny(img.astype(np.uint8), 30, 150)
        edge_mask = cv2.dilate(edges, None, iterations=1)
        edge_mask = cv2.GaussianBlur(edge_mask.astype(float), (5, 5), 2)
        img += edge_mask * 0.15
        
        # 6. Add measurement overlay
        if dilation_cm > 0:
            # Add depth scale markers
            scale_interval = height // 6
            for i in range(6):
                y_pos = i * scale_interval
                cv2.line(img, (width-20, y_pos), (width-10, y_pos), 1.0, 1)
                
            # Add dilation measurement line
            mid_y = int(height * 0.6)
            line_length = int((dilation_cm / 10) * width * 0.5)  # Scale line to dilation
            cv2.line(img, (center_x - line_length, mid_y), 
                    (center_x + line_length, mid_y), 1.0, 1)
        
        # Normalize and convert back to uint8
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def _post_process_image(self, image: Image.Image, dilation_cm: float) -> Image.Image:
        """Apply post-processing to enhance cervical visibility"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply CLAHE with reduced clip limit for more natural contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply bilateral filter for edge-preserving smoothing
        smoothed = cv2.bilateralFilter(enhanced, 9, 50, 50)
        
        # Add ultrasound characteristics
        processed = self._add_ultrasound_characteristics(smoothed, dilation_cm)
        
        # Convert back to RGB
        result_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        
        return Image.fromarray(result_rgb)
        
    def _augment_ultrasound(self, image: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """Apply realistic ultrasound augmentations while preserving anatomical features"""
        if seed is not None:
            np.random.seed(seed)
            
        # Convert to float32 for processing
        img = image.astype(np.float32) / 255.0
        height, width = img.shape[:2]
        
        # 1. Rotation (Probe Movement Simulation)
        angle = np.random.uniform(-20, 20)  # Random rotation ±20°
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
        img = cv2.warpAffine(img, rotation_matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
        
        # 2. Random Flipping
        if np.random.random() < 0.5:
            img = cv2.flip(img, 1)  # Horizontal flip
        
        # 3. Contrast Adjustment
        contrast_factor = np.random.uniform(0.8, 1.2)  # ±20% contrast
        img = np.clip(img * contrast_factor, 0, 1)
        
        # 4. Gaussian Noise (Ultrasound Speckle)
        noise_sigma = np.random.uniform(0.01, 0.05)
        noise = np.random.normal(0, noise_sigma, img.shape)
        img = np.clip(img + noise, 0, 1)
        
        # 5. Elastic Deformation
        dx = gaussian_filter((np.random.rand(height, width) * 2 - 1), sigma=30) * 5
        dy = gaussian_filter((np.random.rand(height, width) * 2 - 1), sigma=30) * 5
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
        img = map_coordinates(img, indices, order=1).reshape(height, width)
        
        # 6. Brightness Adjustment
        brightness_factor = np.random.uniform(0.85, 1.15)  # ±15% brightness
        img = np.clip(img * brightness_factor, 0, 1)
        
        # 7. Random Cropping and Padding
        crop_percent = np.random.uniform(-0.1, 0.1)  # ±10% crop/pad
        crop_size = int(width * (1 + crop_percent))
        if crop_size < width:
            # Crop
            x1 = np.random.randint(0, width - crop_size)
            y1 = np.random.randint(0, height - crop_size)
            img = img[y1:y1+crop_size, x1:x1+crop_size]
            img = cv2.resize(img, (width, height))
        else:
            # Pad
            pad_width = ((0, crop_size - height), (0, crop_size - width))
            img = np.pad(img, pad_width, mode='constant')
            img = cv2.resize(img, (width, height))
        
        # 8. Blur/Sharpening
        if np.random.random() < 0.5:
            # Blur
            blur_radius = np.random.uniform(0.5, 1.5)
            img = cv2.GaussianBlur(img, (0, 0), blur_radius)
        else:
            # Sharpen
            blur = cv2.GaussianBlur(img, (0, 0), 3)
            img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
            
        # 9. Depth-dependent Effects
        depth_factor = np.linspace(1.0, 0.7, height)
        depth_factor = depth_factor.reshape(-1, 1)
        img *= depth_factor
        
        # 10. Curved Probe Effect (preserve/enhance)
        x = np.linspace(-1, 1, width)
        curve = 1 - 0.3 * x**2
        img *= curve.reshape(1, -1)
        
        # Normalize and convert back to uint8
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return img

    def generate_augmented_dataset(self, image_path: str, output_dir: str, num_augmentations: int = 100):
        """Generate augmented versions of a single ultrasound image"""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the original image
        original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        # Resize to 512x512 if needed
        if original.shape != (512, 512):
            original = cv2.resize(original, (512, 512))
        
        # Generate augmented versions
        for i in range(num_augmentations):
            # Apply augmentations with different random seeds
            augmented = self._augment_ultrasound(original, seed=i)
            
            # Save augmented image
            output_path = os.path.join(output_dir, f"aug_{i:03d}.png")
            cv2.imwrite(output_path, augmented)
            
            if (i + 1) % 10 == 0:
                logging.info(f"Generated {i + 1}/{num_augmentations} augmented images")
                
    def generate_images(
        self,
        output_dir: str,
        dilation_cm: float,
        num_images: int = 1,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        strength: float = 0.45  # Reduced to maintain more template features
    ) -> List[str]:
        """Generate a set of synthetic ultrasound images for a given dilation"""
        os.makedirs(output_dir, exist_ok=True)
        generated_paths = []
        
        for i in range(num_images):
            try:
                # Randomly select and preprocess template image
                template = random.choice(self.templates)
                enhanced_template = self._preprocess_template(template)
                
                # Create prompts focusing on cervical visibility
                prompt = self._get_prompt_for_dilation(dilation_cm)
                
                # Generate image with enhanced template
                image = self.img2img(
                    prompt=prompt,
                    image=enhanced_template,
                    strength=strength,  # Lower strength to keep more template features
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                ).images[0]
                
                # Post-process the image
                processed_image = self._post_process_image(image, dilation_cm)
                
                # Save both original and processed versions for comparison
                base_path = os.path.join(output_dir, f"synthetic_dilation_{dilation_cm}cm_{i+1}")
                processed_path = f"{base_path}.png"
                processed_image.save(processed_path, optimize=True)
                generated_paths.append(processed_path)
                
                logger.info(f"Generated image {i+1}/{num_images} for dilation {dilation_cm}cm")
                
            except Exception as e:
                logger.error(f"Failed to generate image {i+1}: {str(e)}")
                continue
            
        return generated_paths
    
    def generate_dataset(
        self,
        output_dir: str,
        num_images_per_dilation: int = 10,
        dilation_range: Optional[List[float]] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        strength: float = 0.75
    ) -> None:
        """
        Generate a dataset of synthetic ultrasound images across different dilations
        
        Args:
            output_dir: Directory to save generated images
            num_images_per_dilation: Number of images to generate per dilation value
            dilation_range: List of dilation values to generate images for
            guidance_scale: Scale for classifier-free guidance
            num_inference_steps: Number of denoising steps
            strength: Amount of noise to add during image-to-image generation
        """
        if dilation_range is None:
            dilation_range = list(range(0, 11, 1))  # 0-10cm in 1cm increments
            
        logger.info(f"Generating dataset with {len(dilation_range)} different dilations")
        
        for dilation in dilation_range:
            logger.info(f"Generating {num_images_per_dilation} images for {dilation}cm dilation")
            self.generate_images(
                output_dir=output_dir,
                dilation_cm=dilation,
                num_images=num_images_per_dilation,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                strength=strength
            )
            
        logger.info("Dataset generation completed!")

if __name__ == "__main__":
    # Example usage with real templates
    generator = UltrasoundGenerator(
        template_dir="/Users/harimanivannan/Documents/GitHub/Cervico/ai/data/images"
    )
    
    # Generate synthetic dataset
    generator.generate_dataset(
        output_dir="/Users/harimanivannan/Documents/GitHub/Cervico/ai/data/synthetic_images",
        num_images_per_dilation=5,
        dilation_range=[2, 4, 6, 8, 10]
    )
