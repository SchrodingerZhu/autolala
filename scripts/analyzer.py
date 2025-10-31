import json
import sys
from pathlib import Path
import islpy as isl
from typing import Any, Tuple, Union, Optional, List, Dict
import argparse
import re
from fractions import Fraction
from collections import defaultdict
import sympy as sp
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import matplotlib.pyplot as plt

# Robust imports for isl polynomial classes across islpy variants
try:
    from islpy import PwQPolynomial as PWQPoly  # preferred name
except Exception:  # pragma: no cover - fallback for older islpy
    from islpy import PiecewiseQuasiPolynomial as PWQPoly  # type: ignore

try:
    from islpy import QPolynomial as QPoly  # preferred name
except Exception:  # pragma: no cover - fallback for older islpy
    from islpy import QuasiPolynomial as QPoly  # type: ignore

class DistItem:
    def __init__(self, qpoly: PWQPoly, cardinality: PWQPoly):
        self.qpoly = qpoly
        self.cardinality = cardinality
    
    @staticmethod
    def _parse_qpoly(s: str) -> PWQPoly:
        # Parse as PWQPoly since the format includes domain mappings
        return DistItem._parse_pwqpoly(s)

    @staticmethod
    def _parse_pwqpoly(s: str) -> PWQPoly:
        # Try direct constructor first
        try:
            return PWQPoly(s)
        except Exception as direct_err:
            # Some versions expose read_from_str on PwQPolynomial
            try:
                read = getattr(PWQPoly, "read_from_str", None)
                if callable(read):
                    return read(isl.DEFAULT_CONTEXT, s)
            except Exception:
                pass
            raise ValueError(f"Failed to parse PwQPolynomial from: {s}\n{direct_err}")
        
    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "DistItem":
        qpoly_str = obj.get("qpoly")
        card_str = obj.get("cardinality")
        if not isinstance(qpoly_str, str) or not isinstance(card_str, str):
            raise TypeError("Each item must contain 'qpoly' and 'cardinality' strings")

        qpoly = cls._parse_qpoly(qpoly_str)
        # Cardinality in the JSON is piecewise
        cardinality = cls._parse_pwqpoly(card_str)

        return cls(qpoly=qpoly, cardinality=cardinality)
    
    def instantiate(self, param: Dict[str, int]) -> "DistItem":
        """
        Instantiate the DistItem by replacing parameters with concrete values.
        
        Process:
        1. Convert ISL structures to strings
        2. Remove "key," patterns (parameter declarations)
        3. Replace remaining keys with their values
        """
        # Process qpoly - use string manipulation as ISL structures
        # follow pattern: [params] -> { expression }
        qpoly_str = str(self.qpoly)
        original_qpoly_str = qpoly_str
        for key, value in param.items():
            # Remove from parameter list: "key," or "key]"
            qpoly_str = re.sub(rf'\b{key}\s*,\s*', '', qpoly_str)
            qpoly_str = re.sub(rf',\s*\b{key}\s*\]', ']', qpoly_str)
            qpoly_str = re.sub(rf'\[\s*\b{key}\s*\]', '[]', qpoly_str)
            # Replace in expressions and constraints
            qpoly_str = re.sub(rf'\b{key}\b', str(value), qpoly_str)
        
        # Process cardinality (piecewise)
        card_str = str(self.cardinality)
        original_card_str = card_str
        for key, value in param.items():
            # Remove from parameter list: "key," or "key]"
            card_str = re.sub(rf'\b{key}\s*,\s*', '', card_str)
            card_str = re.sub(rf',\s*\b{key}\s*\]', ']', card_str)
            card_str = re.sub(rf'\[\s*\b{key}\s*\]', '[]', card_str)
            # Replace in expressions and constraints
            card_str = re.sub(rf'\b{key}\b', str(value), card_str)
        
        # Parse back into ISL structures
        try:
            new_qpoly = self._parse_qpoly(qpoly_str)
        except Exception as e:
            print(f"Failed to parse qpoly after instantiation:", file=sys.stderr)
            print(f"  Original: {original_qpoly_str}", file=sys.stderr)
            print(f"  After substitution: {qpoly_str}", file=sys.stderr)
            raise e
        
        try:
            new_cardinality = self._parse_pwqpoly(card_str)
        except Exception as e:
            print(f"Failed to parse cardinality after instantiation:", file=sys.stderr)
            print(f"  Original: {original_card_str}", file=sys.stderr)
            print(f"  After substitution: {card_str}", file=sys.stderr)
            raise e
        
        return DistItem(qpoly=new_qpoly, cardinality=new_cardinality)

class Distro:
    def __init__(self, total: PWQPoly, items: List[DistItem]):
        self.total = total
        self.items = items

    @staticmethod
    def _parse_total_qpoly(s: str) -> PWQPoly:
        # Parse total as a PwQPolynomial
        return DistItem._parse_pwqpoly(s)

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "Distro":
        total_str = obj.get("total")
        items_list = obj.get("items")
        if not isinstance(total_str, str):
            raise TypeError("Top-level 'total' must be a string")
        if not isinstance(items_list, list):
            raise TypeError("Top-level 'items' must be a list")

        total = cls._parse_total_qpoly(total_str)
        items = [DistItem.from_dict(it) for it in items_list]
        return cls(total=total, items=items)

    @classmethod
    def from_json_file(cls, path: Union[str, Path]) -> "Distro":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"JSON file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise TypeError("Top-level JSON must be an object")
        return cls.from_dict(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the Distro to a dictionary in the input JSON format."""
        return {
            "total": str(self.total),
            "items": [
                {
                    "qpoly": str(item.qpoly),
                    "cardinality": str(item.cardinality)
                }
                for item in self.items
            ]
        }
    
    def to_json_string(self, indent: int = 2) -> str:
        """Convert the Distro to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def to_json_file(self, path: Union[str, Path]) -> None:
        """Save the Distro to a JSON file."""
        p = Path(path)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def instantiate(self, param: Dict[str, int]) -> "Distro":
        """
        Instantiate the Distro by replacing parameters with concrete values.
        
        Process:
        1. Convert ISL structures to strings
        2. Remove "key," patterns (parameter declarations)
        3. Replace remaining keys with their values
        """
        # Process total
        total_str = str(self.total)
        original_total_str = total_str
        for key, value in param.items():
            # Remove from parameter list: "key," or "key]"
            total_str = re.sub(rf'\b{key}\s*,\s*', '', total_str)
            total_str = re.sub(rf',\s*\b{key}\s*\]', ']', total_str)
            total_str = re.sub(rf'\[\s*\b{key}\s*\]', '[]', total_str)
            # Replace in expressions and constraints
            total_str = re.sub(rf'\b{key}\b', str(value), total_str)
        
        # Parse back into ISL structure
        try:
            new_total = self._parse_total_qpoly(total_str)
        except Exception as e:
            print(f"Failed to parse total after instantiation:", file=sys.stderr)
            print(f"  Original: {original_total_str}", file=sys.stderr)
            print(f"  After substitution: {total_str}", file=sys.stderr)
            raise e
        
        # Instantiate all items
        new_items = [item.instantiate(param) for item in self.items]
        
        return Distro(total=new_total, items=new_items)
    
    def drop_r(self) -> "Distro":
        """
        Drop the 'r' parameter from all polynomials in the distribution.
        For total count, this means remove * R part.
        For each item, this means, if any term as * R, remove * R and keep the term.
        Otherwise, remove the term.
        """
        # Process total - remove * R
        total_str = str(self.total)
        original_total_str = total_str
        
        # Remove R from parameter list
        total_str = re.sub(r'\bR\s*,\s*', '', total_str)
        total_str = re.sub(r',\s*\bR\s*\]', ']', total_str)
        total_str = re.sub(r'\[\s*\bR\s*\]', '[]', total_str)
        
        # First, remove * R from expressions (handle various whitespace patterns)
        total_str = re.sub(r'\s*\*\s*R\b', '', total_str)
        total_str = re.sub(r'\bR\s*\*\s*', '', total_str)
        
        # Then replace any remaining standalone R with 1
        total_str = re.sub(r'\bR\b', '1', total_str)
        
        # Remove R >= constraints (after replacing R with 1, clean up constraints like "1 >= 2")
        total_str = re.sub(r'\s+and\s+1\s*>=\s*\d+', '', total_str)
        total_str = re.sub(r'1\s*>=\s*\d+\s+and\s+', '', total_str)
        total_str = re.sub(r'\s*:\s*1\s*>=\s*\d+\s*', '', total_str)
        
        try:
            new_total = self._parse_total_qpoly(total_str)
        except Exception as e:
            print(f"Failed to parse total after dropping R:", file=sys.stderr)
            print(f"  Original: {original_total_str}", file=sys.stderr)
            print(f"  After dropping R: {total_str}", file=sys.stderr)
            raise e
        
        # Process items
        new_items = []
        for item in self.items:
            # Process qpoly
            qpoly_str = str(item.qpoly)
            original_qpoly_str = qpoly_str
            
            # Remove R from parameter list
            qpoly_str = re.sub(r'\bR\s*,\s*', '', qpoly_str)
            qpoly_str = re.sub(r',\s*\bR\s*\]', ']', qpoly_str)
            qpoly_str = re.sub(r'\[\s*\bR\s*\]', '[]', qpoly_str)
            
            # For qpoly, we don't need to remove * R from the expression itself
            # since it shouldn't contain R in the polynomial expression
            
            # Process cardinality - this is where we filter terms
            card_str = str(item.cardinality)
            original_card_str = card_str
            
            # Remove R from parameter list
            card_str = re.sub(r'\bR\s*,\s*', '', card_str)
            card_str = re.sub(r',\s*\bR\s*\]', ']', card_str)
            card_str = re.sub(r'\[\s*\bR\s*\]', '[]', card_str)
            
            # Split into pieces (separated by semicolons in piecewise format)
            # Pattern: expression : constraints; expression : constraints; ...
            pieces = []
            
            # Extract the main content between outermost { }
            match = re.search(r'\{(.+)\}', card_str)
            if match:
                content = match.group(1)
                # Split by semicolon to get individual pieces
                raw_pieces = content.split(';')
                
                for piece in raw_pieces:
                    piece = piece.strip()
                    # Check if this piece contains R in the expression part (before :)
                    if ':' in piece:
                        expr_part, constraint_part = piece.split(':', 1)
                    else:
                        expr_part = piece
                        constraint_part = ''
                    
                    # If expression contains R, remove * R and keep the term
                    if re.search(r'\bR\b', expr_part):
                        # First, remove * R from the expression
                        expr_part = re.sub(r'\s*\*\s*R\b', '', expr_part)
                        expr_part = re.sub(r'\bR\s*\*\s*', '', expr_part)
                        
                        # Then replace any remaining standalone R with 1
                        expr_part = re.sub(r'\bR\b', '1', expr_part)
                        
                        # Remove/update R constraints from constraint part
                        constraint_part = re.sub(r'\s+and\s+R\s*>=\s*\d+', '', constraint_part)
                        constraint_part = re.sub(r'R\s*>=\s*\d+\s+and\s+', '', constraint_part)
                        constraint_part = re.sub(r'^\s*R\s*>=\s*\d+\s*$', '', constraint_part)
                        
                        if constraint_part.strip():
                            pieces.append(f"{expr_part} : {constraint_part}")
                        else:
                            pieces.append(expr_part)
                    # Otherwise, skip this piece (don't include it)
                
                # Reconstruct the cardinality string
                if pieces:
                    # Get the prefix (everything before {)
                    prefix = card_str[:card_str.index('{') + 1]
                    suffix = '}'
                    card_str = prefix + ' ' + '; '.join(pieces) + ' ' + suffix
                else:
                    # No pieces left, create an empty/zero polynomial
                    prefix = card_str[:card_str.index('{') + 1]
                    card_str = prefix + ' 0 }'
            
            try:
                new_qpoly = DistItem._parse_qpoly(qpoly_str)
            except Exception as e:
                print(f"Failed to parse qpoly after dropping R:", file=sys.stderr)
                print(f"  Original: {original_qpoly_str}", file=sys.stderr)
                print(f"  After dropping R: {qpoly_str}", file=sys.stderr)
                raise e
            
            try:
                new_cardinality = DistItem._parse_pwqpoly(card_str)
            except Exception as e:
                print(f"Failed to parse cardinality after dropping R:", file=sys.stderr)
                print(f"  Original: {original_card_str}", file=sys.stderr)
                print(f"  After dropping R: {card_str}", file=sys.stderr)
                raise e
            
            new_items.append(DistItem(qpoly=new_qpoly, cardinality=new_cardinality))
        
        return Distro(total=new_total, items=new_items)

    def to_numeric_distro(self) -> "NumericDistro":
        """
        Convert a fully instantiated Distro to a NumericDistro by evaluating all points.
        The total is just the total value as an integer.
        For each item, let q be the qpoly and c be the cardinality.
        For each piece in c, iterate the points in its domain, evaluate q at that point to get value v as a fraction and increase the counts by the cardinality value
        """
        # Extract total as an integer
        total_str = str(self.total)
        # For a fully instantiated distro, total should be like "{ 31363200 }"
        # Parse out the number
        total_match = re.search(r'\{\s*(\d+)\s*\}', total_str)
        if not total_match:
            raise ValueError(f"Cannot parse total from fully instantiated distro: {total_str}")
        total = int(total_match.group(1))
        
        # Dictionary to count occurrences of each RI value
        counts = defaultdict(Fraction)
        
        # Process each item
        for item_idx, item in enumerate(self.items):
            qpoly_str = str(item.qpoly)
            
            # Collect pieces from cardinality
            # First convert cardinality to set format (remove parameter arrows)
            card_str = str(item.cardinality)
            # Convert "[i2, i3] -> { expr : constraints }" to "{ [i2, i3] : constraints }"
            # This treats the parameters as set dimensions instead
            card_set_str = re.sub(r'\[(.*?)\]\s*->\s*\{\s*(.*?)\s*:\s*(.*?)\s*\}', r'{ [\1] : \3 }', card_str)
            # If no constraints, handle "[i2, i3] -> { expr }" -> "{ [i2, i3] }"
            if card_set_str == card_str:  # No change, try simpler pattern
                card_set_str = re.sub(r'\[(.*?)\]\s*->\s*\{(.*?)\}', r'{ [\1] }', card_str)
            
            # Extract parameter names from the cardinality string
            card_params_match = re.search(r'\[(.*?)\]\s*->', card_str)
            card_param_names = []
            if card_params_match:
                card_param_names = [p.strip() for p in card_params_match.group(1).split(',') if p.strip()]
            
            # Parse as a set to get the domain to iterate
            try:
                card_domain = isl.BasicSet(card_set_str).to_set()
            except:
                # If parsing fails, might be a constant (no parameters)
                card_domain = None
            
            pieces = []
            def collect_piece(qpoly_piece, domain_set):
                # NOTE: ISL swaps these - qpoly_piece is actually the domain/constraints
                # and domain_set is actually the value!
                pieces.append((domain_set, qpoly_piece))
            
            item.cardinality.foreach_piece(collect_piece)
            
            # Process each piece
            for piece_idx, (qpoly_piece, domain_set) in enumerate(pieces):
                card_value_str = str(qpoly_piece)
                domain_str = str(domain_set)
                
                # Debug: print what we got from ISL
                # print(f"DEBUG piece {piece_idx}: value='{card_value_str}', domain='{domain_str}'", file=sys.stderr)
                
                # Extract parameter names from the cardinality piece BEFORE stripping
                # The cardinality is like "[i2] -> { expr }" or just "{ expr }"
                piece_param_names = []
                card_params_match = re.search(r'\[(.*?)\]\s*->', card_value_str)
                if card_params_match and card_params_match.group(1).strip():
                    piece_param_names = [p.strip() for p in card_params_match.group(1).split(',') if p.strip()]
                
                # print(f"DEBUG piece_param_names from value: {piece_param_names}, card_param_names: {card_param_names}", file=sys.stderr)
                
                # Convert the domain from "[i1] -> { : constraints }" to "{ [i1] : constraints }"
                # so we can iterate over it as a set
                if piece_param_names:
                    # Has parameters - need to convert domain
                    # Convert "[i1, i2] -> { : constraints }" to "{ [i1, i2] : constraints }"
                    domain_str_converted = re.sub(r'\[(.*?)\]\s*->\s*\{\s*:\s*(.*?)\s*\}', r'{ [\1] : \2 }', domain_str)
                    # If no constraints, handle "[i1] -> { }" -> "{ [i1] }"
                    if domain_str_converted == domain_str:
                        domain_str_converted = re.sub(r'\[(.*?)\]\s*->\s*\{(.*?)\}', r'{ [\1] }', domain_str)
                    
                    # print(f"DEBUG converted domain: '{domain_str_converted}'", file=sys.stderr)
                    
                    try:
                        piece_domain = isl.BasicSet(domain_str_converted).to_set()
                        # print(f"DEBUG using converted piece domain", file=sys.stderr)
                    except Exception as e:
                        print(f"Warning: Cannot parse converted domain {domain_str_converted}: {e}", file=sys.stderr)
                        piece_domain = None
                elif card_param_names:
                    # Use the global domain and parameters
                    piece_param_names = card_param_names
                    piece_domain = card_domain
                    # print(f"DEBUG using global domain", file=sys.stderr)
                else:
                    # No parameters at all
                    piece_domain = None
                    # print(f"DEBUG no parameters, no domain", file=sys.stderr)
                
                # Remove the parameter arrow part "[i2] -> " if present from the value
                # Convert "[i2] -> { expr }" to "{ expr }"
                card_value_str = re.sub(r'\[.*?\]\s*->\s*', '', card_value_str)
                
                # Parse the cardinality value expression
                # It can be:
                # 1. A constant: "{ 175/2 }" or "{ 99 }"
                # 2. A polynomial: "{ (8925/8 - 175/16 * i2) }"
                # 3. Format with constraints: "{ value : constraints }" (constraints are in domain_set)
                
                # Extract just the expression part (before any ':')
                expr_match = re.search(r'\{\s*([^:}]+)', card_value_str)
                if not expr_match:
                    if '0' not in card_value_str:
                        print(f"Warning: Cannot parse cardinality expression from: {card_value_str}", file=sys.stderr)
                    continue
                
                card_expr_str = expr_match.group(1).strip()
                
                # Remove parentheses if the entire expression is wrapped
                if card_expr_str.startswith('(') and card_expr_str.endswith(')'):
                    card_expr_str = card_expr_str[1:-1].strip()
                
                # Check if this is a constant or polynomial by looking for variables
                # If it contains parameter names from piece_param_names, it's a polynomial
                is_polynomial = any(param in card_expr_str for param in piece_param_names)
                
                if not is_polynomial:
                    # Constant cardinality - parse as fraction
                    try:
                        if '/' in card_expr_str:
                            cardinality_value = Fraction(card_expr_str)
                        else:
                            # Evaluate simple arithmetic if present
                            cardinality_value = Fraction(sp.sympify(card_expr_str))
                    except Exception as e:
                        if '0' not in card_value_str:
                            print(f"Warning: Cannot parse constant cardinality '{card_expr_str}': {e}", file=sys.stderr)
                        continue
                    
                    if cardinality_value == 0:
                        continue
                else:
                    # Polynomial cardinality - will be evaluated point-wise
                    cardinality_value = None
                    card_poly_expr = card_expr_str
                
                # Use the piece_domain we parsed for this piece
                # Parse qpoly to extract variable names and expression
                qpoly_match = re.search(r'\[\s*(.*?)\]\s*->\s*(.+?)\s*\}', qpoly_str)
                
                if piece_domain is None or not piece_param_names:
                    # No parameters - constant cardinality and constant qpoly
                    if cardinality_value is None:
                        print(f"Warning: Polynomial cardinality without domain: {card_value_str}", file=sys.stderr)
                        continue
                    
                    if qpoly_match:
                        expr_str = qpoly_match.group(2).strip()
                        # Evaluate as constant
                        try:
                            import sympy
                            value = sympy.sympify(expr_str)
                            if value.is_number:
                                counts[Fraction(value)] += cardinality_value
                            else:
                                print(f"Warning: Non-numeric constant expression: {expr_str}", file=sys.stderr)
                        except Exception as e:
                            print(f"Warning: Cannot evaluate expression {expr_str}: {e}", file=sys.stderr)
                    else:
                        value = self._evaluate_qpoly_constant(qpoly_str)
                        counts[Fraction(value)] += cardinality_value
                else:
                    # Has parameters - iterate the domain
                    if qpoly_match and qpoly_match.group(1).strip():
                        var_names = [v.strip() for v in qpoly_match.group(1).split(',') if v.strip()]
                        expr_str = qpoly_match.group(2).strip()
                        
                        # Iterate over points in the domain
                        points_list = []
                        try:
                            piece_domain.foreach_point(lambda pt: points_list.append(pt))
                        except Exception as e:
                            # If can't iterate, might be unbounded or error
                            print(f"Warning: Cannot iterate domain {domain_str}: {e}", file=sys.stderr)
                            continue
                        
                        # Evaluate qpoly at each point
                        # Need to build a mapping from piece_param_names to values from point
                        for point in points_list:
                            # Get values for cardinality parameters from point
                            param_values = {}
                            for i, param_name in enumerate(piece_param_names):
                                try:
                                    coord = point.get_coordinate_val(isl.dim_type.set, i)
                                    coord_str = str(coord)
                                    if '/' in coord_str:
                                        param_values[param_name] = Fraction(coord_str)
                                    else:
                                        param_values[param_name] = int(coord_str)
                                except Exception as e:
                                    print(f"Warning: Could not get coordinate {i} for {param_name}: {e}", file=sys.stderr)
                            
                            # Evaluate the cardinality at this point (if polynomial)
                            if cardinality_value is None:
                                # Polynomial cardinality - evaluate at this point
                                try:
                                    card_symbols = {name: sp.Symbol(name) for name in piece_param_names}
                                    card_sympy_expr = sp.sympify(card_poly_expr, locals=card_symbols)
                                    card_result = card_sympy_expr.subs(param_values)
                                    
                                    # Convert to Fraction
                                    if hasattr(card_result, 'p') and hasattr(card_result, 'q'):
                                        point_cardinality = Fraction(int(card_result.p), int(card_result.q))
                                    else:
                                        point_cardinality = Fraction(card_result)
                                    
                                    if point_cardinality == 0:
                                        continue
                                except Exception as e:
                                    print(f"Warning: Could not evaluate cardinality {card_poly_expr} with {param_values}: {e}", file=sys.stderr)
                                    continue
                            else:
                                # Constant cardinality
                                point_cardinality = cardinality_value
                            
                            # Substitute parameter values into the qpoly expression
                            # Use sympy to evaluate
                            try:
                                symbols = {name: sp.Symbol(name) for name in var_names}
                                sympy_expr = sp.sympify(expr_str, locals=symbols)
                                result = sympy_expr.subs(param_values)
                                
                                # Convert to Fraction
                                if hasattr(result, 'p') and hasattr(result, 'q'):
                                    value = Fraction(int(result.p), int(result.q))
                                else:
                                    value = Fraction(result)
                                
                                counts[value] += point_cardinality
                            except Exception as e:
                                print(f"Warning: Could not evaluate {expr_str} with {param_values}: {e}", file=sys.stderr)
                    else:
                        # Constant qpoly but parametric cardinality
                        # Evaluate qpoly once
                        value = self._evaluate_qpoly_constant(qpoly_str)
                        
                        if cardinality_value is None:
                            # Polynomial cardinality - evaluate at each point
                            points_list = []
                            try:
                                piece_domain.foreach_point(lambda pt: points_list.append(pt))
                            except Exception as e:
                                print(f"Warning: Cannot iterate domain {domain_str}: {e}", file=sys.stderr)
                                continue
                            
                            for point in points_list:
                                # Get parameter values for this point
                                param_values = {}
                                for i, param_name in enumerate(piece_param_names):
                                    try:
                                        coord = point.get_coordinate_val(isl.dim_type.set, i)
                                        coord_str = str(coord)
                                        if '/' in coord_str:
                                            param_values[param_name] = Fraction(coord_str)
                                        else:
                                            param_values[param_name] = int(coord_str)
                                    except Exception as e:
                                        print(f"Warning: Could not get coordinate {i} for {param_name}: {e}", file=sys.stderr)
                                
                                # Evaluate cardinality polynomial at this point
                                try:
                                    card_symbols = {name: sp.Symbol(name) for name in piece_param_names}
                                    card_sympy_expr = sp.sympify(card_poly_expr, locals=card_symbols)
                                    card_result = card_sympy_expr.subs(param_values)
                                    
                                    if hasattr(card_result, 'p') and hasattr(card_result, 'q'):
                                        point_cardinality = Fraction(int(card_result.p), int(card_result.q))
                                    else:
                                        point_cardinality = Fraction(card_result)
                                    
                                    if point_cardinality != 0:
                                        counts[Fraction(value)] += point_cardinality
                                except Exception as e:
                                    print(f"Warning: Could not evaluate cardinality {card_poly_expr} with {param_values}: {e}", file=sys.stderr)
                        else:
                            # Constant cardinality - multiply by number of points
                            point_count = 0
                            def count_points(pt):
                                nonlocal point_count
                                point_count += 1
                            try:
                                piece_domain.foreach_point(count_points)
                            except:
                                point_count = 1
                            counts[Fraction(value)] += cardinality_value * point_count
        
        return NumericDistro(total=total, counts=dict(counts))
    
    def _evaluate_qpoly_constant(self, qpoly_str: str) -> Fraction:
        """Extract numeric value from a constant qpoly string."""
        # Match pattern like "{ value }" or "{ [...] -> value }"
        # Value can be a fraction like "312005/2" or integer "156003"
        match = re.search(r'->\s*(-?\d+(?:/\d+)?)\s*\}', qpoly_str)
        if match:
            val_str = match.group(1)
            return Fraction(val_str) if '/' in val_str else Fraction(int(val_str))
        match = re.search(r'\{\s*(-?\d+(?:/\d+)?)\s*\}', qpoly_str)
        if match:
            val_str = match.group(1)
            return Fraction(val_str) if '/' in val_str else Fraction(int(val_str))
        raise ValueError(f"Cannot parse constant qpoly value from: {qpoly_str}")
    
    def _eval_qpoly_at_point(self, qpoly, point, var_names, expr_str):
        """Evaluate a qpoly expression at a specific point."""
        try:
            # Get point coordinates
            var_values = {}
            space = point.get_space()
            
            for i in range(len(var_names)):
                try:
                    # Get coordinate value from point
                    coord_val = point.get_coordinate_val(isl.dim_type.set, i)
                    # Convert to integer/fraction
                    val_str = str(coord_val)
                    # Handle fractions if present
                    if '/' in val_str:
                        var_values[var_names[i]] = Fraction(val_str)
                    else:
                        var_values[var_names[i]] = int(val_str)
                except Exception as e:
                    print(f"Warning: Could not get coordinate {i}: {e}", file=sys.stderr)
                    return None
            
            # Use sympy to evaluate the expression
            symbols = {name: sp.Symbol(name) for name in var_names}
            sympy_expr = sp.sympify(expr_str, locals=symbols)
            result = sympy_expr.subs(var_values)
            
            # Convert to Fraction
            if hasattr(result, 'p') and hasattr(result, 'q'):
                # It's a sympy Rational
                return Fraction(int(result.p), int(result.q))
            else:
                # Try to convert to Fraction
                return Fraction(result)
        except Exception as e:
            print(f"Warning: Could not evaluate {expr_str} at point: {e}", file=sys.stderr)
            return None

class NumericDistro:
    def __init__(self, total: int, counts: Dict[Fraction, Fraction]):
        self.total = total
        self.counts = counts
    
    def _evaluate_qpoly_constant(self, qpoly_str: str) -> int:
        """Extract numeric value from a constant qpoly string."""
        # Match pattern like "{ value }" or "{ [...] -> value }"
        match = re.search(r'->\s*(-?\d+)\s*\}', qpoly_str)
        if match:
            return int(match.group(1))
        match = re.search(r'\{\s*(-?\d+)\s*\}', qpoly_str)
        if match:
            return int(match.group(1))
        raise ValueError(f"Cannot parse constant qpoly value from: {qpoly_str}")
    
    def _eval_qpoly_at_point(self, qpoly, point, var_names, expr_str):
        """Evaluate a qpoly expression at a specific point."""
        try:
            # Get point coordinates
            var_values = {}
            space = point.get_space()
            
            for i in range(len(var_names)):
                try:
                    # Get coordinate value from point
                    coord_val = point.get_coordinate_val(isl.dim_type.set, i)
                    # Convert to integer
                    val_str = str(coord_val)
                    # Handle fractions if present
                    if '/' in val_str:
                        var_values[var_names[i]] = Fraction(val_str)
                    else:
                        var_values[var_names[i]] = int(val_str)
                except Exception as e:
                    print(f"Warning: Could not get coordinate {i}: {e}", file=sys.stderr)
                    return None
            
            # Use sympy to evaluate the expression
            symbols = {name: sp.Symbol(name) for name in var_names}
            sympy_expr = sp.sympify(expr_str, locals=symbols)
            result = sympy_expr.subs(var_values)
            
            # Convert to int
            return int(result)
        except Exception as e:
            print(f"Warning: Could not evaluate {expr_str} at point: {e}", file=sys.stderr)
            return None

class NumericDistro:
    def __init__(self, total: int, counts: Dict[Fraction, Fraction]):
        self.total = total
        # Sort by RI value and store as list of (ri_value, count) tuples
        self.counts = [(float(x), float(y)) for x, y in sorted(list(counts.items()))]
        
        # Precompute miss_ratio and turning_points (following Rust implementation)
        # miss_ratio[i] = cumulative probability of RI values >= counts[i]
        # turning_points[i] = cumulative reuse distance up to index i
        
        # First compute probabilities from counts
        probabilities = [count / total for _, count in self.counts]
        
        # Compute miss_ratio (cumulative sum from end to start)
        # miss_ratio[i] = sum of probabilities[j] for all j >= i
        self.miss_ratio = [0.0] * len(probabilities)
        rolling_sum = max(1.0 - sum(probabilities), 0)
        for i in range(len(probabilities) - 1, -1, -1):
            rolling_sum += probabilities[i]
            self.miss_ratio[i] = rolling_sum
        
        # Compute turning_points
        self.turning_points = [0.0] * len(self.counts)
        prev = 0.0
        for i in range(1, len(self.counts)):
            ri_diff = self.counts[i][0] - self.counts[i-1][0]
            self.turning_points[i] = prev + self.miss_ratio[i-1] * ri_diff
            prev = self.turning_points[i]

    def miss_ratio_for(self, cache_size: float) -> float:
        """
        Binary search the turning point. Return miss_ratio * total at that index.
        """
        # Binary search to find the largest index where turning_points[i] <= cache_size
        left, right = 0, len(self.turning_points) - 1
        result_idx = -1
        
        while left <= right:
            mid = (left + right) // 2
            if self.turning_points[mid] <= cache_size:
                result_idx = mid
                left = mid + 1
            else:
                right = mid - 1
        
        # If no turning point is <= cache_size, all accesses miss
        if result_idx == -1:
            return 1.0
        
        # Return miss_ratio * total at the found index
        return self.miss_ratio[result_idx]

def _print_summary(d: Distro):
    # Lightweight summary that does not evaluate polynomials
    print("Deserialized distribution summary:")
    # Print space tuple dims for orientation
    print("- total: QPolynomial")
    print(f"- items: {len(d.items)} entries")


def _compute_miss_ratio(args):
    """Worker function for parallel processing of miss ratio computation."""
    p0, distro = args
    import math
    instantiated = distro.instantiate({'p0': p0, 'p1': p0})
    numeric_distro = instantiated.to_numeric_distro()
    c = p0 / 8
    miss_ratio = numeric_distro.miss_ratio_for(cache_size=c)
    return (p0, miss_ratio)



def main(argv: Optional[List[str]] = None) -> int:
    import math
    parser = argparse.ArgumentParser(description="Deserialize an ISL distribution JSON file.")
    parser.add_argument("json_path", help="Path to the JSON file to load.")
    parser.add_argument("--summary", action="store_true", help="Print a brief summary after loading.")
    parser.add_argument("--output", "-o", default="miss_count_plot.png", help="Output file for the plot.")
    args = parser.parse_args(argv)
    try:
        distro = Distro.from_json_file(args.json_path)
        distro = distro.drop_r()
        
        # Prepare arguments for parallel processing
        p0_values = list(range(10, 401, 1))
        worker_args = [(p0, distro) for p0 in p0_values]
        
        # Parallel processing with tqdm
        with Pool(processes=cpu_count()) as pool:
            results = list(tqdm(
                pool.imap(_compute_miss_ratio, worker_args, chunksize=1),
                total=len(worker_args),
                desc="Computing miss ratios"
            ))
        
        # Sort results by p0 (should already be sorted, but just to be safe)
        results.sort(key=lambda x: x[0])
        
        # Extract p0 and miss_count for plotting
        p0_vals = [r[0] for r in results]
        miss_ratios = [r[1] for r in results]

        # Create and save plot
        plt.figure(figsize=(10, 6))
        plt.plot(p0_vals, miss_ratios, marker='o', linestyle='-', linewidth=2, markersize=4)
        plt.xlabel('p0', fontsize=12)
        plt.ylabel('Miss Ratio', fontsize=12)
        plt.title('Miss Ratio vs p0', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {args.output}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.summary:
        _print_summary(distro)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
