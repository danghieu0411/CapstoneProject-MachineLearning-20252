import re
import sys

def check_latex_syntax(filepath):
    print(f"Checking {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Check brace matching
    stack = []
    lines = content.split('\n')
    brace_errors = []
    
    # Simple checker ignoring escaped braces \{ and \}
    # Also ignoring inside verbatim or lstlisting environments
    in_verbatim = False
    in_lstlisting = False
    
    for i, line in enumerate(lines, 1):
        # Strip comments
        # check if we enter/exit verbatim or lstlisting
        if '\\begin{verbatim}' in line:
            in_verbatim = True
            continue
        if '\\end{verbatim}' in line:
            in_verbatim = False
            continue
        if '\\begin{lstlisting}' in line:
            in_lstlisting = True
            continue
        if '\\end{lstlisting}' in line:
            in_lstlisting = False
            continue
            
        if in_verbatim or in_lstlisting:
            continue
            
        # Ignore comments starting with % (if not escaped)
        # Find comments
        comment_idx = -1
        for idx in range(len(line)):
            if line[idx] == '%' and (idx == 0 or line[idx-1] != '\\'):
                comment_idx = idx
                break
        if comment_idx != -1:
            line = line[:comment_idx]
            
        for char_idx, char in enumerate(line):
            if char == '{':
                if char_idx > 0 and line[char_idx-1] == '\\':
                    continue  # escaped brace
                stack.append((i, char_idx))
            elif char == '}':
                if char_idx > 0 and line[char_idx-1] == '\\':
                    continue  # escaped brace
                if not stack:
                    brace_errors.append((i, f"Extra closing brace '}}' at line {i}, col {char_idx}"))
                else:
                    stack.pop()

    for item in stack:
        brace_errors.append((item[0], f"Unmatched opening brace '{{' at line {item[0]}, col {item[1]}"))

    if brace_errors:
        print("Found Brace Errors:")
        for err in brace_errors:
            print(f"  Line {err[0]}: {err[1]}")
    else:
        print("No brace matching errors found.")

    # 2. Check \begin and \end matching
    env_stack = []
    env_errors = []
    in_verbatim = False
    in_lstlisting = False
    
    for i, line in enumerate(lines, 1):
        # We need to extract environment names
        begins = re.findall(r'\\begin\{([^}]+)\}', line)
        ends = re.findall(r'\\end\{([^}]+)\}', line)
        
        # Note: we need to handle nested environments and order
        # But usually they are one per line or well-spaced.
        # Let's do a simple check
        for b in begins:
            env_stack.append((i, b))
        for e in ends:
            if not env_stack:
                env_errors.append((i, f"Unmatched \\end{{{e}}} at line {i}"))
            else:
                last_line, last_env = env_stack.pop()
                if last_env != e:
                    env_errors.append((i, f"Environment mismatch: started with \\begin{{{last_env}}} on line {last_line}, closed with \\end{{{e}}} on line {i}"))

    for item in env_stack:
        env_errors.append((item[0], f"Unmatched \\begin{{{item[1]}}} on line {item[0]}"))

    if env_errors:
        print("\nFound Environment Errors:")
        for err in env_errors:
            print(f"  {err[1]}")
    else:
        print("No environment matching errors found.")

    # 3. Check for raw underscores outside math mode or listings/verbatim
    in_verbatim = False
    in_lstlisting = False
    in_math = False
    math_block = False
    underscore_errors = []
    
    for i, line in enumerate(lines, 1):
        if '\\begin{verbatim}' in line:
            in_verbatim = True
            continue
        if '\\end{verbatim}' in line:
            in_verbatim = False
            continue
        if '\\begin{lstlisting}' in line:
            in_lstlisting = True
            continue
        if '\\end{lstlisting}' in line:
            in_lstlisting = False
            continue
            
        if in_verbatim or in_lstlisting:
            continue
            
        # Strip comments
        comment_idx = -1
        for idx in range(len(line)):
            if line[idx] == '%' and (idx == 0 or line[idx-1] != '\\'):
                comment_idx = idx
                break
        if comment_idx != -1:
            line = line[:comment_idx]

        # Ignore lines that contain hyperref target links or package loads
        if '\\usepackage' in line or '\\href' in line or '\\includegraphics' in line or '\\label' in line:
            continue

        # Check for inline math block transitions
        # We also have to be careful with double dollar signs $$
        # But we can check character by character
        char_idx = 0
        while char_idx < len(line):
            char = line[char_idx]
            if char == '$':
                in_math = not in_math
            elif char == '_':
                # Check if it is escaped: \_
                if char_idx > 0 and line[char_idx-1] == '\\':
                    pass
                elif not in_math:
                    # We might be in a equation environment
                    # Let's check if there is an equation/align block active in general, 
                    # but simple line check: is there a raw underscore?
                    # Let's report it to inspect manually
                    # Wait, we can see if the line is inside an equation block by checking if we are inside begin{equation}
                    # We didn't track equation block in this simple loop, but we can check it
                    underscore_errors.append((i, char_idx, line.strip()))
            char_idx += 1

    # Filter out lines that are equation blocks
    actual_underscore_errors = []
    in_equation = False
    for line_num, col, text in underscore_errors:
        # Check if line_num is inside \begin{equation} or similar
        # Let's scan from beginning up to line_num
        in_eq = False
        for l_idx in range(line_num):
            l_text = lines[l_idx]
            if '\\begin{equation}' in l_text or '\\begin{align}' in l_text:
                in_eq = True
            if '\\end{equation}' in l_text or '\\end{align}' in l_text:
                in_eq = False
        if not in_eq:
            actual_underscore_errors.append((line_num, col, text))

    if actual_underscore_errors:
        print("\nFound Raw Underscores (outside math/verbatim/listings):")
        for err in actual_underscore_errors[:20]:
            print(f"  Line {err[0]}, col {err[1]}: '{err[2]}'")
    else:
        print("No raw underscore errors found.")

if __name__ == '__main__':
    files = [
        '/home/duy/Downloads/CapstoneProject-MachineLearning-20252/report/Report.tex',
        '/home/duy/Downloads/CapstoneProject-MachineLearning-20252/report/Report.md',
        '/home/duy/Downloads/CapstoneProject-MachineLearning-20252/README.md'
    ]
    for filepath in files:
        check_latex_syntax(filepath)
        print("-" * 50)

