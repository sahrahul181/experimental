import os
import sys

def process_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    doc_comments = []
    has_written_header = False
    
    with open(output_path, 'w', encoding='utf-8') as out_f:
        i = 0
        while i < len(lines):
            raw_line = lines[i]
            line = raw_line.strip()
            
            if line.startswith("///"):
                doc_text = line[3:]
                if doc_text.startswith(" "):
                    doc_text = doc_text[1:]
                doc_comments.append(doc_text)
            elif line.startswith("pub "):
                if doc_comments:
                    if not has_written_header:
                        out_f.write(f"# API Reference: {input_path}\n\n")
                        has_written_header = True
                    
                    signature = [line]
                    current_line = line
                    
                    i += 1
                    while i < len(lines):
                        if current_line.endswith(";") or current_line.endswith("{") or "{" in current_line or ";" in current_line:
                            break
                        next_line = lines[i].strip()
                        signature.append(next_line)
                        current_line = next_line
                        i += 1
                    i -= 1 # adjust for the main loop increment

                    final_sig = " ".join(signature).rstrip(" {;")
                    out_f.write(f"### `{final_sig}`\n\n")
                    for doc in doc_comments:
                        out_f.write(f"{doc}\n")
                    out_f.write("\n---\n\n")
                doc_comments = []
            else:
                if len(line) > 0:
                    doc_comments = []
            
            i += 1
            
        if not has_written_header:
            # File had no documented pub declarations, we can remove the empty markdown file
            pass

def main():
    if len(sys.argv) < 3:
        print("Usage: python generate_llm_docs.py <input_dir> <output_dir>")
        sys.exit(1)
        
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    os.makedirs(output_dir, exist_ok=True)
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.zig'):
                input_path = os.path.join(root, file)
                
                # compute relative path and replace separators with underscore
                rel_path = os.path.relpath(input_path, input_dir)
                md_name = rel_path.replace(os.sep, "_").replace("/", "_")
                if md_name.endswith('.zig'):
                    md_name = md_name[:-4] + ".md"
                    
                output_path = os.path.join(output_dir, md_name)
                
                try:
                    process_file(input_path, output_path)
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")
                    
    # cleanup empty markdown files
    for file in os.listdir(output_dir):
        if file.endswith('.md'):
            file_path = os.path.join(output_dir, file)
            if os.path.getsize(file_path) == 0:
                os.remove(file_path)
                
    print("Documentation generation complete.")

if __name__ == "__main__":
    main()
