import re

def sync():
    # Read the clean, validated Report.tex
    with open('/home/duy/Downloads/CapstoneProject-MachineLearning-20252/report/Report.tex', 'r', encoding='utf-8') as f:
        tex_content = f.read()

    # Replace unicode box-drawing characters with standard ASCII equivalents
    tex_content = tex_content.replace('│', '|').replace('▼', 'v')
    
    # Save the cleaned content back to Report.tex
    with open('/home/duy/Downloads/CapstoneProject-MachineLearning-20252/report/Report.tex', 'w', encoding='utf-8') as f:
        f.write(tex_content)
    print("Cleaned and updated report/Report.tex")

    # 1. Write to report/Report.md (exact copy)
    with open('/home/duy/Downloads/CapstoneProject-MachineLearning-20252/report/Report.md', 'w', encoding='utf-8') as f:
        f.write(tex_content)
    print("Synchronized report/Report.md")

    # 2. Prepend 'report/' to image paths for README.md
    # We find all \includegraphics{filename} or \includegraphics[...]{filename}
    # and replace with \includegraphics{report/filename} or \includegraphics[...]{report/filename}
    
    def replace_image_path(match):
        options = match.group(1) if match.group(1) else ""
        filename = match.group(2)
        # If the filename already starts with 'report/', don't prepend again
        if filename.startswith('report/'):
            return f"\\includegraphics{options}{{{filename}}}"
        else:
            return f"\\includegraphics{options}{{{'report/' + filename}}}"

    readme_content = re.sub(r'\\includegraphics(\[[^\]]+\])?\{([^}]+)\}', replace_image_path, tex_content)
    
    with open('/home/duy/Downloads/CapstoneProject-MachineLearning-20252/README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print("Synchronized README.md")

if __name__ == '__main__':
    sync()
