#!/usr/bin/env python3
"""
Simple icon generator for browser extension
Creates PNG icons with gradient background and shield emoji
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_icon(size, filename):
    """Create a square icon with gradient and emoji"""
    # Create image with gradient
    img = Image.new('RGB', (size, size), '#667eea')
    draw = ImageDraw.Draw(img)
    
    # Draw gradient (simple version)
    for i in range(size):
        color = (
            int(102 + (118 - 102) * i / size),  # R
            int(126 + (75 - 126) * i / size),   # G
            int(234 + (162 - 234) * i / size)   # B
        )
        draw.rectangle([(0, i), (size, i+1)], fill=color)
    
    # Draw white circle in center
    margin = size // 4
    draw.ellipse(
        [(margin, margin), (size-margin, size-margin)],
        fill='white',
        outline='white'
    )
    
    # Try to add text (shield emoji or text)
    try:
        font_size = size // 2
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        text = "🛡"
        
        # Get text bbox
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center text
        x = (size - text_width) // 2
        y = (size - text_height) // 2 - font_size // 4
        
        draw.text((x, y), text, fill='#667eea', font=font)
    except:
        # Fallback: just draw a shield shape
        points = [
            (size//2, size//4),
            (size*3//4, size*2//5),
            (size*3//4, size*3//5),
            (size//2, size*3//4),
            (size//4, size*3//5),
            (size//4, size*2//5),
        ]
        draw.polygon(points, fill='#667eea')
    
    # Save
    img.save(filename, 'PNG')
    print(f"✅ Created {filename} ({size}x{size})")

# Create icons directory
os.makedirs('icons', exist_ok=True)

# Generate icons
create_icon(16, 'icons/icon16.png')
create_icon(48, 'icons/icon48.png')
create_icon(128, 'icons/icon128.png')

print("\n🎉 All icons created successfully!")
