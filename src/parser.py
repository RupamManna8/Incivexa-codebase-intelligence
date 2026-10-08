import ast
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

try:
    from tree_sitter import Language, Parser
    import tree_sitter_javascript
    JS_LANGUAGE = Language(tree_sitter_javascript.language())
    JS_PARSER = Parser(JS_LANGUAGE)
except Exception:
    JS_PARSER = None

@dataclass
class CodeSymbol:
    name: str
    symbol_type: str  # "function", "method", "class"
    file_path: str
    start_line: int   # 1-indexed
    end_line: int     # 1-indexed
    parameters: List[str]
    calls: List[str]
    code_snippet: str
    docstring: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CodeParser:
    @staticmethod
    def parse_python(file_path: str, content: str) -> List[CodeSymbol]:
        """Parses Python code using the standard library ast module."""
        symbols: List[CodeSymbol] = []
        lines = content.splitlines()
        try:
            tree = ast.parse(content, filename=file_path)
        except SyntaxError:
            return symbols

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = node.lineno
                end = getattr(node, "end_lineno", start)
                snippet = "\n".join(lines[start - 1 : end])
                params = [arg.arg for arg in node.args.args]
                calls = [
                    call.func.id
                    for call in ast.walk(node)
                    if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                ]
                docstring = ast.get_docstring(node)
                symbols.append(
                    CodeSymbol(
                        name=node.name,
                        symbol_type="function",
                        file_path=file_path,
                        start_line=start,
                        end_line=end,
                        parameters=params,
                        calls=list(set(calls)),
                        code_snippet=snippet,
                        docstring=docstring
                    )
                )
            elif isinstance(node, ast.ClassDef):
                start = node.lineno
                end = getattr(node, "end_lineno", start)
                snippet = "\n".join(lines[start - 1 : min(start + 20, len(lines))])
                symbols.append(
                    CodeSymbol(
                        name=node.name,
                        symbol_type="class",
                        file_path=file_path,
                        start_line=start,
                        end_line=end,
                        parameters=[],
                        calls=[],
                        code_snippet=snippet,
                        docstring=ast.get_docstring(node)
                    )
                )
        return symbols

    @staticmethod
    def parse_javascript(file_path: str, content: str) -> List[CodeSymbol]:
        """Parses JavaScript/TypeScript using tree-sitter."""
        symbols: List[CodeSymbol] = []
        if JS_PARSER is None:
            return symbols

        content_bytes = content.encode("utf-8")
        tree = JS_PARSER.parse(content_bytes)
        lines = content.splitlines()

        def get_text(node) -> str:
            return content_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

        def visit(node):
            if node.type in ["function_declaration", "generator_function_declaration"]:
                name_node = node.child_by_field_name("name")
                name = get_text(name_node) if name_node else "anonymous"
                start = node.start_point[0] + 1
                end = node.end_point[0] + 1
                snippet = "\n".join(lines[start - 1 : end])
                symbols.append(
                    CodeSymbol(
                        name=name,
                        symbol_type="function",
                        file_path=file_path,
                        start_line=start,
                        end_line=end,
                        parameters=[],
                        calls=[],
                        code_snippet=snippet
                    )
                )
            elif node.type == "method_definition":
                name_node = node.child_by_field_name("name")
                name = get_text(name_node) if name_node else "anonymous"
                start = node.start_point[0] + 1
                end = node.end_point[0] + 1
                snippet = "\n".join(lines[start - 1 : end])
                symbols.append(
                    CodeSymbol(
                        name=name,
                        symbol_type="method",
                        file_path=file_path,
                        start_line=start,
                        end_line=end,
                        parameters=[],
                        calls=[],
                        code_snippet=snippet
                    )
                )
            elif node.type == "class_declaration":
                name_node = node.child_by_field_name("name")
                name = get_text(name_node) if name_node else "anonymous"
                start = node.start_point[0] + 1
                end = node.end_point[0] + 1
                snippet = "\n".join(lines[start - 1 : min(start + 25, len(lines))])
                symbols.append(
                    CodeSymbol(
                        name=name,
                        symbol_type="class",
                        file_path=file_path,
                        start_line=start,
                        end_line=end,
                        parameters=[],
                        calls=[],
                        code_snippet=snippet
                    )
                )
            elif node.type == "lexical_declaration" or node.type == "variable_declaration":
                # Check for const fn = () => {}
                for child in node.children:
                    if child.type == "variable_declarator":
                        name_node = child.child_by_field_name("name")
                        val_node = child.child_by_field_name("value")
                        if val_node and val_node.type in ["arrow_function", "function_expression"]:
                            name = get_text(name_node) if name_node else "anonymous"
                            start = child.start_point[0] + 1
                            end = child.end_point[0] + 1
                            snippet = "\n".join(lines[start - 1 : end])
                            symbols.append(
                                CodeSymbol(
                                    name=name,
                                    symbol_type="function",
                                    file_path=file_path,
                                    start_line=start,
                                    end_line=end,
                                    parameters=[],
                                    calls=[],
                                    code_snippet=snippet
                                )
                            )

            for child in node.children:
                visit(child)

        visit(tree.root_node)
        return symbols

    @classmethod
    def parse_file(cls, file_path: str, content: str) -> List[CodeSymbol]:
        if file_path.endswith(".py"):
            return cls.parse_python(file_path, content)
        elif file_path.endswith((".js", ".mjs", ".ts", ".jsx", ".tsx")):
            return cls.parse_javascript(file_path, content)
        return []

    @staticmethod
    def find_enclosing_symbol(symbols: List[CodeSymbol], line_number: int) -> Optional[CodeSymbol]:
        """Finds the narrowest function/method symbol enclosing the given line number."""
        best: Optional[CodeSymbol] = None
        min_span = float("inf")

        for sym in symbols:
            if sym.start_line <= line_number <= sym.end_line:
                span = sym.end_line - sym.start_line
                if span < min_span:
                    min_span = span
                    best = sym
        return best

    @classmethod
    def extract_chunks_for_indexing(cls, file_path: str, content: str) -> List[Dict[str, Any]]:
        """Extracts searchable code blocks/chunks from a file for vector indexing."""
        symbols = cls.parse_file(file_path, content)
        chunks: List[Dict[str, Any]] = []

        if symbols:
            for s in symbols:
                chunks.append({
                    "file_path": file_path,
                    "symbol_name": s.name,
                    "symbol_type": s.symbol_type,
                    "start_line": s.start_line,
                    "end_line": s.end_line,
                    "text": f"File: {file_path}\nType: {s.symbol_type} {s.name}\nCode:\n{s.code_snippet}"
                })
        else:
            # Fallback: line-window chunking
            lines = content.splitlines()
            window_size = 40
            step = 30
            for i in range(0, len(lines), step):
                chunk_lines = lines[i : i + window_size]
                if not chunk_lines:
                    continue
                start_l = i + 1
                end_l = min(i + window_size, len(lines))
                chunks.append({
                    "file_path": file_path,
                    "symbol_name": f"{file_path}:{start_l}-{end_l}",
                    "symbol_type": "block",
                    "start_line": start_l,
                    "end_line": end_l,
                    "text": f"File: {file_path} (lines {start_l}-{end_l})\n" + "\n".join(chunk_lines)
                })

        return chunks
