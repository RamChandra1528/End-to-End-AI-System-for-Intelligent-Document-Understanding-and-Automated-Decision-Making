"""
Explainability Module for Model Interpretability
Supports attention visualization, SHAP, and LIME
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import cv2
from loguru import logger
from pathlib import Path


class AttentionVisualizer:
    """Visualize attention patterns from transformer models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.resolution = tuple(config.get('heatmap_resolution', [800, 600]))
    
    def visualize_attention(
        self,
        image: Image.Image,
        attentions: List[np.ndarray],
        words: List[str],
        boxes: List[List[int]],
        output_path: str,
        layer: int = -1
    ) -> str:
        """
        Visualize attention patterns on document image
        
        Args:
            image: Original document image
            attentions: List of attention matrices from each layer
            words: OCR words
            boxes: Bounding boxes for words
            output_path: Where to save visualization
            layer: Which attention layer to visualize (-1 for last)
        """
        if not attentions:
            logger.warning("No attention data provided")
            return None
        
        # Get attention from specified layer
        attention = attentions[layer]  # Shape: [num_heads, seq_len, seq_len]
        
        # Average across attention heads
        attention_avg = attention.mean(axis=0)  # Shape: [seq_len, seq_len]
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Attention heatmap
        sns.heatmap(
            attention_avg[:50, :50],  # Show first 50 tokens for clarity
            ax=axes[0],
            cmap='YlOrRd',
            cbar=True
        )
        axes[0].set_title('Attention Heatmap')
        axes[0].set_xlabel('Key Tokens')
        axes[0].set_ylabel('Query Tokens')
        
        # Plot 2: Attention overlay on image
        img_with_attention = self._overlay_attention_on_image(
            image, attention_avg, boxes
        )
        axes[1].imshow(img_with_attention)
        axes[1].axis('off')
        axes[1].set_title('Attention Overlay on Document')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Attention visualization saved to: {output_path}")
        return output_path
    
    def _overlay_attention_on_image(
        self,
        image: Image.Image,
        attention: np.ndarray,
        boxes: List[List[int]]
    ) -> np.ndarray:
        """Overlay attention scores on document image"""
        # Convert PIL to numpy
        img_array = np.array(image.convert('RGB'))
        
        # Create attention overlay
        overlay = img_array.copy()
        
        # Get attention scores for each word (average of its attention)
        word_attention = attention.mean(axis=0)  # Average attention received by each token
        
        # Normalize
        if word_attention.max() > 0:
            word_attention = word_attention / word_attention.max()
        
        # Draw bounding boxes with attention-based color
        for i, box in enumerate(boxes[:len(word_attention)]):
            if i >= len(word_attention):
                break
            
            attention_score = word_attention[i]
            
            # Skip if box is invalid
            if len(box) != 4:
                continue
            
            x1, y1, x2, y2 = box
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, img_array.shape[1]))
            x2 = max(0, min(x2, img_array.shape[1]))
            y1 = max(0, min(y1, img_array.shape[0]))
            y2 = max(0, min(y2, img_array.shape[0]))
            
            # Color intensity based on attention
            color = (int(255 * attention_score), 0, 0)  # Red channel
            
            # Draw semi-transparent rectangle
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        
        # Blend original and overlay
        result = cv2.addWeighted(img_array, 0.6, overlay, 0.4, 0)
        
        return result
    
    def create_token_importance_plot(
        self,
        words: List[str],
        importance_scores: np.ndarray,
        output_path: str,
        top_k: int = 20
    ) -> str:
        """Create bar plot of token importance"""
        # Get top-k tokens
        top_indices = np.argsort(importance_scores)[-top_k:][::-1]
        top_words = [words[i] if i < len(words) else f"token_{i}" for i in top_indices]
        top_scores = importance_scores[top_indices]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_words)), top_scores, color='steelblue')
        plt.yticks(range(len(top_words)), top_words)
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_k} Important Tokens')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path


class SHAPExplainer:
    """SHAP-based explainability for model predictions"""
    
    def __init__(self, model, config: Dict):
        self.model = model
        self.config = config
        self.top_k = config.get('top_k_features', 10)
    
    def explain_prediction(
        self,
        inputs: Dict[str, torch.Tensor],
        prediction: int,
        output_path: str
    ) -> Dict:
        """
        Generate SHAP explanations for a prediction
        
        Note: This is a simplified implementation.
        Full SHAP for transformers requires significant computation.
        """
        try:
            import shap
            
            # For demonstration, we'll use gradient-based attribution
            # which is more efficient than full SHAP
            self.model.eval()
            
            # Enable gradients for inputs
            input_ids = inputs['input_ids'].clone().requires_grad_(True)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=inputs['attention_mask'],
                bbox=inputs.get('bbox', torch.zeros_like(input_ids).unsqueeze(-1).repeat(1, 1, 4)),
                pixel_values=inputs['pixel_values']
            )
            
            # Get prediction logit
            target_logit = outputs['class_logits'][0, prediction]
            
            # Backward pass
            target_logit.backward()
            
            # Get gradients as importance
            importance = input_ids.grad[0].abs().sum(dim=-1).cpu().numpy()
            
            # Normalize
            if importance.max() > 0:
                importance = importance / importance.max()
            
            return {
                'importance_scores': importance,
                'method': 'gradient',
                'top_k_indices': np.argsort(importance)[-self.top_k:][::-1].tolist()
            }
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return {'importance_scores': np.array([]), 'method': 'failed'}


class ExplainabilityEngine:
    """Unified explainability interface"""
    
    def __init__(self, model, config: Dict):
        self.model = model
        self.config = config
        self.attention_viz = AttentionVisualizer(config)
        self.shap_explainer = SHAPExplainer(model, config) if config.get('use_shap', False) else None
        self.output_dir = Path(config.get('output_dir', 'outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def explain(
        self,
        image: Image.Image,
        text: str,
        words: List[str],
        boxes: List[List[int]],
        prediction: Dict,
        document_id: str
    ) -> Dict[str, str]:
        """
        Generate comprehensive explanations for a prediction
        
        Returns:
            Dictionary with paths to generated visualizations
        """
        explanations = {}
        
        # 1. Attention visualization
        if self.config.get('use_attention', True) and 'attentions' in prediction:
            attention_path = self.output_dir / f"{document_id}_attention.png"
            try:
                path = self.attention_viz.visualize_attention(
                    image=image,
                    attentions=prediction['attentions'],
                    words=words,
                    boxes=boxes,
                    output_path=str(attention_path)
                )
                explanations['attention_map'] = path
            except Exception as e:
                logger.error(f"Attention visualization failed: {e}")
        
        # 2. Token importance
        if 'reasoning_features' in prediction:
            # Use reasoning features as proxy for importance
            reasoning_features = prediction['reasoning_features']
            importance = np.abs(reasoning_features)
            
            # Expand to match number of words (simplified)
            if len(importance) < len(words):
                importance = np.pad(importance, (0, len(words) - len(importance)))
            else:
                importance = importance[:len(words)]
            
            importance_path = self.output_dir / f"{document_id}_importance.png"
            try:
                path = self.attention_viz.create_token_importance_plot(
                    words=words,
                    importance_scores=importance,
                    output_path=str(importance_path)
                )
                explanations['token_importance'] = path
            except Exception as e:
                logger.error(f"Token importance plot failed: {e}")
        
        # 3. SHAP explanation
        if self.shap_explainer and self.config.get('use_shap', False):
            try:
                # Prepare inputs
                inputs = self.model.prepare_inputs(text, image, words, boxes)
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                shap_result = self.shap_explainer.explain_prediction(
                    inputs=inputs,
                    prediction=prediction['document_type'],
                    output_path=str(self.output_dir / f"{document_id}_shap.png")
                )
                
                explanations['shap_values'] = shap_result
            except Exception as e:
                logger.error(f"SHAP explanation failed: {e}")
        
        # 4. Create summary visualization
        summary_path = self.output_dir / f"{document_id}_summary.png"
        self._create_summary_visualization(
            image=image,
            prediction=prediction,
            words=words,
            boxes=boxes,
            output_path=str(summary_path)
        )
        explanations['summary'] = str(summary_path)
        
        return explanations
    
    def _create_summary_visualization(
        self,
        image: Image.Image,
        prediction: Dict,
        words: List[str],
        boxes: List[List[int]],
        output_path: str
    ) -> None:
        """Create a summary visualization with predictions and key regions"""
        # Create figure
        fig = plt.figure(figsize=(16, 10))
        
        # Plot image
        ax = plt.subplot(111)
        ax.imshow(image)
        ax.axis('off')
        
        # Add prediction text
        doc_type_map = {0: 'Invoice', 1: 'Resume', 2: 'Report'}
        doc_type = doc_type_map.get(prediction.get('document_type', 0), 'Unknown')
        confidence = prediction.get('document_type_confidence', 0.0)
        
        text_str = f"Prediction: {doc_type}\nConfidence: {confidence:.2%}"
        ax.text(
            10, 30, text_str,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=14, fontweight='bold', color='darkblue'
        )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Summary visualization saved to: {output_path}")


def highlight_important_regions(
    image: Image.Image,
    boxes: List[List[int]],
    importance_scores: List[float],
    output_path: str,
    threshold: float = 0.5
) -> str:
    """Highlight important regions in document"""
    # Convert to cv2 format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    overlay = img_cv.copy()
    
    # Draw important boxes
    for box, score in zip(boxes, importance_scores):
        if score > threshold:
            x1, y1, x2, y2 = box
            
            # Color based on importance
            intensity = int(255 * score)
            color = (0, intensity, 0)  # Green channel
            
            # Draw rectangle
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Add score label
            label = f"{score:.2f}"
            cv2.putText(
                overlay, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
    
    # Blend
    result = cv2.addWeighted(img_cv, 0.7, overlay, 0.3, 0)
    
    # Save
    cv2.imwrite(output_path, result)
    
    return output_path
