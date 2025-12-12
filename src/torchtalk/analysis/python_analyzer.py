"""Python module analyzer for PyTorch source code."""

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .helpers import truncate

log = logging.getLogger(__name__)


@dataclass
class PyFunction:
    """Python function or method definition."""

    name: str
    qualified_name: str  # module.class.method or module.function
    file_path: str
    line_number: int
    is_method: bool = False
    is_async: bool = False
    decorators: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    signature: Optional[str] = None
    # For methods, track the class
    class_name: Optional[str] = None
    # C++ binding if detected (e.g., calls torch._C.*)
    cpp_binding: Optional[str] = None


@dataclass
class PyClass:
    """Python class definition."""

    name: str
    qualified_name: str  # module.ClassName
    file_path: str
    line_number: int
    bases: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    methods: List[PyFunction] = field(default_factory=list)
    # For torch.nn.Module subclasses
    is_module: bool = False


@dataclass
class PyImport:
    """Python import statement."""

    module: str  # What's being imported from
    name: str  # What's being imported
    alias: Optional[str] = None  # as X
    file_path: str = ""
    line_number: int = 0


@dataclass
class PyModule:
    """Analyzed Python module."""

    name: str  # e.g., torch.nn.modules.linear
    file_path: str
    classes: List[PyClass] = field(default_factory=list)
    functions: List[PyFunction] = field(default_factory=list)
    imports: List[PyImport] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)  # __all__


class PythonAnalyzer:
    """Analyzes Python source code using AST."""

    def __init__(self):
        self._module_cache: Dict[str, PyModule] = {}

    def analyze_file(self, file_path: str) -> Optional[PyModule]:
        """Analyze a single Python file."""
        path = Path(file_path)
        if not path.exists() or not path.suffix == ".py":
            return None

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(content, filename=str(path))
        except SyntaxError as e:
            log.warning(f"Syntax error in {file_path}: {e}")
            return None

        # Derive module name from path
        module_name = self._path_to_module_name(path)

        module = PyModule(name=module_name, file_path=str(path))

        # Extract components
        visitor = _ASTVisitor(module, content)
        visitor.visit(tree)

        self._module_cache[module_name] = module
        return module

    def analyze_directory(
        self, directory: str, pattern: str = "**/*.py", skip_tests: bool = True
    ) -> Dict[str, PyModule]:
        """Analyze all Python files in a directory."""
        dir_path = Path(directory)
        modules: Dict[str, PyModule] = {}

        files = list(dir_path.glob(pattern))
        log.info(f"Analyzing {len(files)} Python files in {directory}...")

        for py_file in files:
            # Skip test files and __pycache__
            rel_path = str(py_file.relative_to(dir_path))
            if skip_tests and ("/test" in rel_path or "test_" in py_file.name):
                continue
            if "__pycache__" in rel_path:
                continue

            module = self.analyze_file(str(py_file))
            if module:
                modules[module.name] = module

        log.info(f"Analyzed {len(modules)} modules")
        return modules

    def _path_to_module_name(self, path: Path) -> str:
        """Convert file path to Python module name."""
        # Try to find torch/ or similar package root
        parts = path.parts
        for i, part in enumerate(parts):
            if part in ("torch", "torchvision", "torchaudio"):
                # Build module name from this point
                module_parts = list(parts[i:])
                # Remove .py extension
                if module_parts[-1].endswith(".py"):
                    module_parts[-1] = module_parts[-1][:-3]
                # Handle __init__.py
                if module_parts[-1] == "__init__":
                    module_parts = module_parts[:-1]
                return ".".join(module_parts)

        # Fallback: just use filename
        return path.stem


class _ASTVisitor(ast.NodeVisitor):
    """AST visitor to extract module components."""

    def __init__(self, module: PyModule, content: str):
        self.module = module
        self.content = content
        self._current_class: Optional[PyClass] = None

    def visit_Import(self, node: ast.Import):
        """Handle: import x, import x as y"""
        for alias in node.names:
            imp = PyImport(
                module=alias.name,
                name=alias.name,
                alias=alias.asname,
                file_path=self.module.file_path,
                line_number=node.lineno,
            )
            self.module.imports.append(imp)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Handle: from x import y, from x import y as z"""
        module_name = node.module or ""
        for alias in node.names:
            imp = PyImport(
                module=module_name,
                name=alias.name,
                alias=alias.asname,
                file_path=self.module.file_path,
                line_number=node.lineno,
            )
            self.module.imports.append(imp)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        """Handle class definitions."""
        bases = [self._get_name(base) for base in node.bases]

        # Check if it's a torch.nn.Module subclass
        is_module = any(b in ("Module", "nn.Module", "torch.nn.Module") for b in bases)

        py_class = PyClass(
            name=node.name,
            qualified_name=f"{self.module.name}.{node.name}",
            file_path=self.module.file_path,
            line_number=node.lineno,
            bases=bases,
            decorators=[self._get_name(d) for d in node.decorator_list],
            docstring=ast.get_docstring(node),
            is_module=is_module,
        )

        # Visit methods
        old_class = self._current_class
        self._current_class = py_class

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._visit_function(item, is_method=True)

        self._current_class = old_class
        self.module.classes.append(py_class)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Handle function definitions at module level."""
        if self._current_class is None:
            self._visit_function(node, is_method=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Handle async function definitions."""
        if self._current_class is None:
            self._visit_function(node, is_method=False, is_async=True)

    def _visit_function(
        self, node: ast.FunctionDef, is_method: bool, is_async: bool = False
    ):
        """Process a function/method definition."""
        # Build signature
        sig = self._build_signature(node)

        # Check for C++ bindings in function body
        cpp_binding = self._find_cpp_binding(node)

        if is_method and self._current_class:
            qualified_name = f"{self._current_class.qualified_name}.{node.name}"
            class_name = self._current_class.name
        else:
            qualified_name = f"{self.module.name}.{node.name}"
            class_name = None

        func = PyFunction(
            name=node.name,
            qualified_name=qualified_name,
            file_path=self.module.file_path,
            line_number=node.lineno,
            is_method=is_method,
            is_async=is_async or isinstance(node, ast.AsyncFunctionDef),
            decorators=[self._get_name(d) for d in node.decorator_list],
            docstring=ast.get_docstring(node),
            signature=sig,
            class_name=class_name,
            cpp_binding=cpp_binding,
        )

        if is_method and self._current_class:
            self._current_class.methods.append(func)
        else:
            self.module.functions.append(func)

    def visit_Assign(self, node: ast.Assign):
        """Handle assignments, looking for __all__."""
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                if isinstance(node.value, ast.List):
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            self.module.exports.append(elt.value)
        self.generic_visit(node)

    def _get_name(self, node: ast.expr) -> str:
        """Get string representation of an AST node (for names, attributes)."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return self._get_name(node.func)
        elif isinstance(node, ast.Subscript):
            return self._get_name(node.value)
        elif isinstance(node, ast.Constant):
            return str(node.value)
        return ""

    def _build_signature(self, node: ast.FunctionDef) -> str:
        """Build function signature string."""
        args = []

        # Regular args
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {self._get_name(arg.annotation)}"
            args.append(arg_str)

        # *args
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")

        # **kwargs
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")

        sig = f"({', '.join(args)})"

        # Return type
        if node.returns:
            sig += f" -> {self._get_name(node.returns)}"

        return truncate(sig, 100)

    def _find_cpp_binding(self, node: ast.FunctionDef) -> Optional[str]:
        """Look for C++ binding calls in function body."""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func_name = self._get_name(child.func)
                # Common C++ binding patterns
                if any(
                    pattern in func_name
                    for pattern in ["torch._C", "torch.ops.", "_C.", "torch.ops.aten"]
                ):
                    return func_name
        return None


def build_module_index(modules: Dict[str, PyModule]) -> Dict[str, List]:
    """Build searchable indexes from analyzed modules."""
    by_class: Dict[str, List[PyClass]] = {}
    by_function: Dict[str, List[PyFunction]] = {}
    by_export: Dict[str, str] = {}  # export name -> module name
    nn_modules: List[PyClass] = []

    for module in modules.values():
        # Index classes
        for cls in module.classes:
            by_class.setdefault(cls.name, []).append(cls)
            if cls.is_module:
                nn_modules.append(cls)

        # Index functions
        for func in module.functions:
            by_function.setdefault(func.name, []).append(func)

        # Index exports
        for export in module.exports:
            by_export[export] = module.name

    return {
        "by_class": by_class,
        "by_function": by_function,
        "by_export": by_export,
        "nn_modules": nn_modules,
    }
