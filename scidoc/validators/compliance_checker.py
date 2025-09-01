"""
Compliance checker for SciDoc.

This module provides validation rules and compliance checking
for project files and metadata.
"""

from typing import Dict, List, Any

from ..models import ValidationResult, ProjectMetadata


class ComplianceChecker:
    """Checks project compliance against validation rules."""
    
    def __init__(self, rules: List[Dict[str, Any]]):
        """
        Initialize compliance checker.
        
        Args:
            rules: List of validation rules
        """
        self.rules = rules
    
    def validate_project(self, metadata: ProjectMetadata) -> List[ValidationResult]:
        """
        Validate project against all rules.
        
        Args:
            metadata: Project metadata
            
        Returns:
            List of validation results
        """
        results = []
        
        for rule in self.rules:
            rule_results = self._apply_rule(metadata, rule)
            results.extend(rule_results)
        
        return results
    
    def _apply_rule(self, metadata: ProjectMetadata, rule: Dict[str, Any]) -> List[ValidationResult]:
        """Apply a single validation rule."""
        results = []
        rule_name = rule.get("name", "unknown")
        
        # Check required files
        if "required_files" in rule:
            required_files = rule["required_files"]
            for required_file in required_files:
                found = any(f.name == required_file for f in metadata.files)
                if not found:
                    results.append(ValidationResult(
                        filename=required_file,
                        rule_name=rule_name,
                        passed=False,
                        message=f"Required file '{required_file}' not found",
                        severity="error",
                        suggestions=[f"Create file '{required_file}'"]
                    ))
        
        # Check file patterns
        if "file_patterns" in rule:
            for pattern_rule in rule["file_patterns"]:
                pattern = pattern_rule.get("pattern", "")
                max_size = pattern_rule.get("max_size")
                min_size = pattern_rule.get("min_size")
                
                matching_files = [f for f in metadata.files if f.name.endswith(pattern)]
                
                for file_meta in matching_files:
                    if max_size and file_meta.size > max_size:
                        results.append(ValidationResult(
                            filename=file_meta.filename,
                            rule_name=rule_name,
                            passed=False,
                            message=f"File size {file_meta.size} exceeds maximum {max_size}",
                            severity="warning"
                        ))
                    
                    if min_size and file_meta.size < min_size:
                        results.append(ValidationResult(
                            filename=file_meta.filename,
                            rule_name=rule_name,
                            passed=False,
                            message=f"File size {file_meta.size} below minimum {min_size}",
                            severity="warning"
                        ))
        
        return results
