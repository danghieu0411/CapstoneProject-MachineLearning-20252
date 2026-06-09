import math
from PIL import Image, ImageDraw, ImageFont

def create_pipeline_diagram():
    # Base image: high resolution
    width = 1600
    height = 500
    im = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(im)

    # Styling colors
    BOX_BORDER = (41, 128, 185, 255)     # #2980b9 (Strong blue)
    BOX_FILL = (235, 245, 251, 255)       # #ebf5fb (Very light blue)
    TEXT_COLOR = (44, 62, 80, 255)       # #2c3e50 (Dark slate)
    ARROW_COLOR = (41, 128, 185, 255)    # Matching blue for cohesive design
    SHADOW_COLOR = (220, 220, 220, 255)  # Light gray

    # Load font
    font_path = "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"
    try:
        font = ImageFont.truetype(font_path, 18)
    except IOError:
        font = ImageFont.load_default()

    def draw_box(x1, y1, x2, y2, text):
        # Draw soft shadow
        draw.rounded_rectangle((x1 + 4, y1 + 4, x2 + 4, y2 + 4), radius=15, fill=SHADOW_COLOR)
        # Draw main box
        draw.rounded_rectangle((x1, y1, x2, y2), radius=15, fill=BOX_FILL, outline=BOX_BORDER, width=3)
        
        # Center the text
        lines = text.split('\n')
        total_height = 0
        line_sizes = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            line_sizes.append((w, h))
            total_height += h + 8
        total_height -= 8
        
        curr_y = y1 + (y2 - y1 - total_height) / 2
        for line, (w, h) in zip(lines, line_sizes):
            curr_x = x1 + (x2 - x1 - w) / 2
            draw.text((curr_x, curr_y - 2), line, fill=TEXT_COLOR, font=font)
            curr_y += h + 8

    def draw_arrow(start, end, width=4, head_size=18):
        x1, y1 = start
        x2, y2 = end
        draw.line([start, end], fill=ARROW_COLOR, width=width)
        
        dx = x2 - x1
        dy = y2 - y1
        angle = math.atan2(dy, dx)
        
        arrow_p1 = (x2 - head_size * math.cos(angle - math.pi/6), y2 - head_size * math.sin(angle - math.pi/6))
        arrow_p2 = (x2 - head_size * math.cos(angle + math.pi/6), y2 - head_size * math.sin(angle + math.pi/6))
        
        draw.polygon([end, arrow_p1, arrow_p2], fill=ARROW_COLOR)

    # Box coordinates
    # Row 0: y in [60, 180]
    # Row 1: y in [320, 440]
    # Cols:
    # 0: [50, 370]
    # 1: [430, 750]
    # 2: [810, 1130]
    # 3: [1190, 1510]
    boxes = [
        # Row 0 (Left-to-Right)
        (50, 60, 370, 180, "Raw Images\n(JPEG)"),
        (430, 60, 750, 180, "Grayscale & Resize\n(256 x 256)"),
        (810, 60, 1130, 180, "SIFT\nKeypoints & Descriptors"),
        (1190, 60, 1510, 180, "K-Means Vocabulary\n(k = 500)"),
        
        # Row 1 (Right-to-Left, following flow)
        (1190, 320, 1510, 440, "BoVW Histogram\n(L2-norm)"),
        (810, 320, 1130, 440, "Train/Test Split\n(80 / 20)"),
        (430, 320, 750, 440, "Standard Scaler"),
        (50, 320, 370, 440, "Random Forest\n(100 trees)")
    ]

    # Draw boxes
    for x1, y1, x2, y2, text in boxes:
        draw_box(x1, y1, x2, y2, text)

    # Draw connections
    # Box 0 -> Box 1
    draw_arrow((370, 120), (430, 120))
    # Box 1 -> Box 2
    draw_arrow((750, 120), (810, 120))
    # Box 2 -> Box 3
    draw_arrow((1130, 120), (1190, 120))
    # Box 3 -> Box 4 (down arrow)
    draw_arrow((1350, 180), (1350, 320))
    # Box 4 -> Box 5 (left arrow)
    draw_arrow((1190, 380), (1130, 380))
    # Box 5 -> Box 6 (left arrow)
    draw_arrow((810, 380), (750, 380))
    # Box 6 -> Box 7 (left arrow)
    draw_arrow((430, 380), (370, 380))

    # Save output
    output_path = "/home/duy/Downloads/CapstoneProject-MachineLearning-20252/report/sift_bovw_pipeline.png"
    im.save(output_path, "PNG")
    print(f"Successfully drew pipeline diagram and saved to {output_path}")

if __name__ == "__main__":
    create_pipeline_diagram()
