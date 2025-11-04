"""
AI Reasoning Layer for Intelligent Decision-Making
Validates, ranks, and detects anomalies in extracted data
"""
import re
from typing import Dict, List, Optional, Tuple
from loguru import logger
import numpy as np
from datetime import datetime


class ReasoningEngine:
    """AI Reasoning engine for document validation and decision-making"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.invoice_config = config.get('invoice', {})
        self.resume_config = config.get('resume', {})
        self.report_config = config.get('report', {})
    
    def validate_invoice(self, extracted_fields: Dict) -> Dict:
        """
        Validate invoice data and check for inconsistencies
        """
        result = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'validation_details': {}
        }
        
        # Check required fields
        required_fields = self.invoice_config.get('check_required_fields', [])
        missing_fields = [f for f in required_fields if f not in extracted_fields or not extracted_fields[f]]
        
        if missing_fields:
            result['valid'] = False
            result['issues'].append(f"Missing required fields: {', '.join(missing_fields)}")
            result['validation_details']['missing_fields'] = missing_fields
        
        # Validate total calculation
        if self.invoice_config.get('validate_totals', True):
            validation = self._validate_invoice_totals(extracted_fields)
            result['validation_details']['total_validation'] = validation
            
            if not validation['valid']:
                result['valid'] = False
                result['issues'].append(validation['message'])
            elif validation.get('warning'):
                result['warnings'].append(validation['warning'])
        
        # Validate date format
        if 'date' in extracted_fields:
            date_valid = self._validate_date(extracted_fields['date'])
            result['validation_details']['date_valid'] = date_valid
            if not date_valid:
                result['warnings'].append("Date format could not be validated")
        
        # Check for duplicate invoice number
        if 'invoice_no' in extracted_fields:
            result['validation_details']['invoice_no'] = extracted_fields['invoice_no']
        
        # Anomaly detection
        anomalies = self._detect_invoice_anomalies(extracted_fields)
        if anomalies:
            result['warnings'].extend(anomalies)
            result['validation_details']['anomalies'] = anomalies
        
        return result
    
    def rank_resumes(self, resumes: List[Dict], job_requirements: Optional[Dict] = None) -> List[Dict]:
        """
        Rank resumes based on criteria
        """
        ranked_resumes = []
        
        for resume in resumes:
            score = self._calculate_resume_score(resume, job_requirements)
            ranked_resumes.append({
                **resume,
                'score': score['total_score'],
                'score_breakdown': score['breakdown'],
                'rank': 0  # Will be filled after sorting
            })
        
        # Sort by score (descending)
        ranked_resumes.sort(key=lambda x: x['score'], reverse=True)
        
        # Assign ranks
        for i, resume in enumerate(ranked_resumes):
            resume['rank'] = i + 1
        
        return ranked_resumes
    
    def validate_report(self, extracted_fields: Dict) -> Dict:
        """
        Validate report structure and completeness
        """
        result = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'validation_details': {}
        }
        
        # Check required sections
        required_sections = self.report_config.get('check_sections', [])
        found_sections = extracted_fields.get('sections', [])
        
        missing_sections = [s for s in required_sections if not any(s.lower() in found.lower() for found in found_sections)]
        
        if missing_sections:
            result['warnings'].append(f"Missing recommended sections: {', '.join(missing_sections)}")
            result['validation_details']['missing_sections'] = missing_sections
        
        # Check word count
        min_word_count = self.report_config.get('min_word_count', 100)
        word_count = extracted_fields.get('word_count', 0)
        
        if word_count < min_word_count:
            result['warnings'].append(f"Word count ({word_count}) is below recommended minimum ({min_word_count})")
        
        result['validation_details']['word_count'] = word_count
        result['validation_details']['sections_found'] = found_sections
        
        return result
    
    def _validate_invoice_totals(self, fields: Dict) -> Dict:
        """Validate invoice total calculations"""
        line_items = fields.get('line_items', [])
        declared_total = fields.get('total_amount')
        
        if not line_items or not declared_total:
            return {
                'valid': True,
                'message': 'Insufficient data to validate totals'
            }
        
        # Calculate expected total
        calculated_total = sum(item.get('quantity', 0) * item.get('price', 0) for item in line_items)
        
        # Parse declared total
        try:
            declared_value = float(str(declared_total).replace(',', '').replace('INR', '').replace('USD', '').strip())
        except ValueError:
            return {
                'valid': False,
                'message': f"Could not parse total amount: {declared_total}"
            }
        
        # Compare with tolerance
        tolerance = self.invoice_config.get('tolerance', 0.01)
        difference = abs(calculated_total - declared_value)
        relative_diff = difference / max(declared_value, 1)
        
        if relative_diff > tolerance:
            return {
                'valid': False,
                'message': f"Total mismatch: Calculated={calculated_total:.2f}, Declared={declared_value:.2f}, Difference={difference:.2f}",
                'calculated_total': calculated_total,
                'declared_total': declared_value,
                'difference': difference
            }
        elif relative_diff > tolerance / 2:
            return {
                'valid': True,
                'warning': f"Minor discrepancy in totals (within tolerance): Difference={difference:.2f}",
                'calculated_total': calculated_total,
                'declared_total': declared_value,
                'difference': difference
            }
        else:
            return {
                'valid': True,
                'message': 'Total validated successfully',
                'calculated_total': calculated_total,
                'declared_total': declared_value
            }
    
    def _validate_date(self, date_str: str) -> bool:
        """Validate date format"""
        date_patterns = [
            r'\d{2}/\d{2}/\d{4}',
            r'\d{4}-\d{2}-\d{2}',
            r'\d{2}-\d{2}-\d{4}',
            r'[A-Za-z]+ \d{1,2}, \d{4}'
        ]
        
        for pattern in date_patterns:
            if re.search(pattern, date_str):
                return True
        
        return False
    
    def _detect_invoice_anomalies(self, fields: Dict) -> List[str]:
        """Detect anomalies in invoice data"""
        anomalies = []
        
        # Check for unusually high amounts
        total = fields.get('total_amount')
        if total:
            try:
                amount = float(str(total).replace(',', '').replace('INR', '').replace('USD', '').strip())
                if amount > 1000000:  # More than 1 million
                    anomalies.append(f"Unusually high amount: {amount}")
                if amount <= 0:
                    anomalies.append(f"Invalid amount: {amount}")
            except:
                pass
        
        # Check for duplicate line items
        line_items = fields.get('line_items', [])
        descriptions = [item.get('description', '') for item in line_items]
        if len(descriptions) != len(set(descriptions)):
            anomalies.append("Duplicate line items detected")
        
        # Check for missing vendor
        if not fields.get('vendor'):
            anomalies.append("Vendor information not found")
        
        return anomalies
    
    def _calculate_resume_score(self, resume: Dict, job_requirements: Optional[Dict] = None) -> Dict:
        """Calculate resume score based on criteria"""
        criteria = self.resume_config.get('ranking_criteria', [])
        scores = {}
        weights = {
            'experience_years': 0.3,
            'education_level': 0.25,
            'skills_match': 0.35,
            'certifications': 0.1
        }
        
        # Experience score
        experience = resume.get('experience_years', 0)
        min_exp = self.resume_config.get('min_experience', 0)
        
        if experience >= min_exp + 10:
            scores['experience_years'] = 100
        elif experience >= min_exp + 5:
            scores['experience_years'] = 80
        elif experience >= min_exp:
            scores['experience_years'] = 60
        else:
            scores['experience_years'] = max(0, 40 - (min_exp - experience) * 10)
        
        # Education score
        education = resume.get('education', [])
        education_scores = {
            'phd': 100,
            'master': 85,
            'mba': 80,
            'm.tech': 85,
            'm.sc': 85,
            'bachelor': 70,
            'b.tech': 70,
            'b.sc': 70
        }
        
        if education:
            max_edu_score = max([education_scores.get(e.lower(), 50) for e in education])
            scores['education_level'] = max_edu_score
        else:
            scores['education_level'] = 50
        
        # Skills match score
        skills = resume.get('skills', [])
        required_skills = job_requirements.get('required_skills', []) if job_requirements else []
        
        if required_skills:
            matched_skills = [s for s in skills if any(req.lower() in s.lower() for req in required_skills)]
            match_percentage = len(matched_skills) / len(required_skills) * 100
            scores['skills_match'] = min(100, match_percentage)
        else:
            # Base score on number of skills
            scores['skills_match'] = min(100, len(skills) * 5)
        
        # Certifications score
        certifications = resume.get('certifications', [])
        scores['certifications'] = min(100, len(certifications) * 25)
        
        # Calculate weighted total
        total_score = sum(scores.get(criterion, 0) * weights.get(criterion, 0) for criterion in criteria)
        
        return {
            'total_score': round(total_score, 2),
            'breakdown': scores
        }
    
    def make_decision(self, document_type: str, extracted_fields: Dict, validation_result: Dict) -> Dict:
        """
        Make final decision based on document type and validation
        """
        decision = {
            'decision': 'Unknown',
            'confidence': 0.0,
            'reasoning': []
        }
        
        if document_type == 'invoice':
            if validation_result['valid'] and not validation_result['issues']:
                decision['decision'] = 'Valid'
                decision['confidence'] = 0.95 if not validation_result['warnings'] else 0.85
                decision['reasoning'].append("All validations passed")
            elif validation_result['issues']:
                decision['decision'] = 'Invalid'
                decision['confidence'] = 0.90
                decision['reasoning'].extend(validation_result['issues'])
            else:
                decision['decision'] = 'Review Required'
                decision['confidence'] = 0.70
                decision['reasoning'].extend(validation_result['warnings'])
        
        elif document_type == 'resume':
            score = extracted_fields.get('score', 0)
            if score >= 80:
                decision['decision'] = 'Highly Recommended'
                decision['confidence'] = 0.90
            elif score >= 60:
                decision['decision'] = 'Recommended'
                decision['confidence'] = 0.75
            elif score >= 40:
                decision['decision'] = 'Consider'
                decision['confidence'] = 0.60
            else:
                decision['decision'] = 'Not Recommended'
                decision['confidence'] = 0.50
            
            decision['reasoning'].append(f"Resume score: {score}/100")
        
        elif document_type == 'report':
            if validation_result['valid'] and not validation_result['warnings']:
                decision['decision'] = 'Complete'
                decision['confidence'] = 0.90
                decision['reasoning'].append("Report meets all requirements")
            else:
                decision['decision'] = 'Needs Revision'
                decision['confidence'] = 0.70
                decision['reasoning'].extend(validation_result['warnings'])
        
        return decision


class RuleBasedReasoning:
    """Rule-based reasoning for specific business logic"""
    
    def __init__(self):
        self.rules = {
            'invoice': [
                self._rule_invoice_must_have_number,
                self._rule_invoice_amount_positive,
                self._rule_invoice_date_not_future
            ],
            'resume': [
                self._rule_resume_must_have_contact,
                self._rule_resume_experience_reasonable
            ]
        }
    
    def apply_rules(self, document_type: str, fields: Dict) -> List[Dict]:
        """Apply business rules and return violations"""
        violations = []
        
        rules = self.rules.get(document_type, [])
        for rule in rules:
            violation = rule(fields)
            if violation:
                violations.append(violation)
        
        return violations
    
    def _rule_invoice_must_have_number(self, fields: Dict) -> Optional[Dict]:
        if not fields.get('invoice_no'):
            return {
                'rule': 'invoice_must_have_number',
                'severity': 'error',
                'message': 'Invoice number is required'
            }
        return None
    
    def _rule_invoice_amount_positive(self, fields: Dict) -> Optional[Dict]:
        total = fields.get('total_amount')
        if total:
            try:
                amount = float(str(total).replace(',', '').replace('INR', '').replace('USD', '').strip())
                if amount <= 0:
                    return {
                        'rule': 'invoice_amount_positive',
                        'severity': 'error',
                        'message': f'Invoice amount must be positive, got {amount}'
                    }
            except:
                pass
        return None
    
    def _rule_invoice_date_not_future(self, fields: Dict) -> Optional[Dict]:
        # Simplified check - would need proper date parsing
        date_str = fields.get('date', '')
        if 'future' in date_str.lower():  # Placeholder logic
            return {
                'rule': 'invoice_date_not_future',
                'severity': 'warning',
                'message': 'Invoice date appears to be in the future'
            }
        return None
    
    def _rule_resume_must_have_contact(self, fields: Dict) -> Optional[Dict]:
        if not fields.get('email') and not fields.get('phone'):
            return {
                'rule': 'resume_must_have_contact',
                'severity': 'error',
                'message': 'Resume must have at least email or phone contact'
            }
        return None
    
    def _rule_resume_experience_reasonable(self, fields: Dict) -> Optional[Dict]:
        experience = fields.get('experience_years', 0)
        if experience > 50:
            return {
                'rule': 'resume_experience_reasonable',
                'severity': 'warning',
                'message': f'Experience of {experience} years seems unreasonable'
            }
        return None
