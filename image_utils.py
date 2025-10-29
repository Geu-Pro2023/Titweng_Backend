from PIL import Image, ImageDraw, ImageFont
import os
from datetime import datetime

def add_watermark(image_path: str, cow_tag: str, output_path: str) -> str:
    """Add Titweng watermark to cow facial image"""
    try:
        # Open the image
        with Image.open(image_path) as img:
            # Convert to RGBA for transparency
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Create a transparent overlay
            overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Watermark text
            date_str = datetime.now().strftime("%d-%b-%Y")
            watermark_text = f"TITWENG VERIFIED\n{cow_tag}\n{date_str}"
            
            # Try to use a font, fallback to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            # Calculate text size and position (bottom-right corner)
            bbox = draw.textbbox((0, 0), watermark_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = img.width - text_width - 20
            y = img.height - text_height - 20
            
            # Draw semi-transparent background
            padding = 10
            draw.rectangle([
                x - padding, y - padding,
                x + text_width + padding, y + text_height + padding
            ], fill=(0, 0, 0, 128))
            
            # Draw white text
            draw.multiline_text((x, y), watermark_text, font=font, fill=(255, 255, 255, 255))
            
            # Combine original image with overlay
            watermarked = Image.alpha_composite(img, overlay)
            
            # Convert back to RGB and save
            watermarked = watermarked.convert('RGB')
            watermarked.save(output_path, 'JPEG', quality=85)
            
            return output_path
            
    except Exception as e:
        print(f"Watermarking failed: {e}")
        # If watermarking fails, just copy the original
        with Image.open(image_path) as img:
            img.save(output_path, 'JPEG', quality=85)
        return output_path

def save_facial_image(image_bytes: bytes, cow_id: int, cow_tag: str) -> str:
    """Save and watermark cow facial image"""
    try:
        # Create directory if it doesn't exist
        os.makedirs("static/cow_faces", exist_ok=True)
        
        # Temporary path for original image
        temp_path = f"static/cow_faces/temp_{cow_id}.jpg"
        final_path = f"static/cow_faces/cow_{cow_id}_face.jpg"
        
        # Save original image temporarily
        with open(temp_path, "wb") as f:
            f.write(image_bytes)
        
        # Add watermark and save final image
        watermarked_path = add_watermark(temp_path, cow_tag, final_path)
        
        # Remove temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return final_path
        
    except Exception as e:
        print(f"Failed to save facial image: {e}")
        return None