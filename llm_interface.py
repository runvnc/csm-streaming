import re
import os
from typing import List, Dict, Any, Optional
from openai import OpenAI

class LLMInterface:
    def __init__(self, model_path: str, max_tokens: int = 8192, n_threads: int = 8, gpu_layers: int = -1):
        """Initialize the LLM interface using OpenAI-compatible API.
        
        Args:
            model_path (str): Model name for OpenAI API, or model identifier for local server
            max_tokens (int, optional): Maximum context length. Defaults to 8192.
            n_threads (int, optional): Not used, maintained for API compatibility.
            gpu_layers (int, optional): Not used, maintained for API compatibility.
        
        Environment variables:
            OPENAI_API_KEY: API key for OpenAI (required for OpenAI API)
            OPENAI_BASE_URL: Base URL for OpenAI-compatible server (optional, for local servers)
        """
        # Get configuration from environment
        api_key = os.environ.get("OPENAI_API_KEY", "sk-no-key-required")
        base_url = os.environ.get("OPENAI_BASE_URL", None)
        
        # Initialize OpenAI client
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)
        
        # Store configuration
        self.model = model_path
        self.config = {
            "model_path": model_path,
            "max_tokens": max_tokens,
        }
        
    def trim_to_last_sentence(self, text: str) -> str:        
        """
        Return *text* truncated at the final full sentence boundary.
        A boundary is considered to be any '.', '!' or '?' followed by
        optional quotes/brackets, optional whitespace, and then end-of-string.

        If no sentence terminator exists, the original text is returned.
        """
        # Regex explanation:
        #   (.*?[.!?]["')\]]?)   any text lazily until a terminator
        #   \s*$                 followed only by whitespace till end-of-string
        m = re.match(r"^(.*?[.!?][\"')\]]?)\s*$", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        # Fall back to manual search (handles cases with additional text)
        for i in range(len(text) - 1, -1, -1):
            if text[i] in ".!?":
                return text[: i + 1].strip()
        return text.strip()
    
    def generate_response(self, system_prompt: str, user_message: str, conversation_history: str = "") -> str:
        """Generate a response from the LLM using chat-style prompt formatting.
        
        Args:
            system_prompt (str): The system prompt/instructions
            user_message (str): The user's input message
            conversation_history (str, optional): Any prior conversation context. Defaults to "".
            
        Returns:
            str: The generated response
        """
        # Build messages list
        messages = [{"role": "system", "content": system_prompt}]
        
        # Parse conversation history if provided
        if conversation_history:
            # Try to parse the history format "User: ...\nAI: ..."
            lines = conversation_history.strip().split('\n')
            current_role = None
            current_content = []
            
            for line in lines:
                if line.startswith("User: "):
                    if current_role and current_content:
                        messages.append({"role": current_role, "content": '\n'.join(current_content)})
                    current_role = "user"
                    current_content = [line[6:]]  # Remove "User: " prefix
                elif line.startswith("AI: "):
                    if current_role and current_content:
                        messages.append({"role": current_role, "content": '\n'.join(current_content)})
                    current_role = "assistant"
                    current_content = [line[4:]]  # Remove "AI: " prefix
                elif current_role:
                    current_content.append(line)
            
            # Add the last message if any
            if current_role and current_content:
                messages.append({"role": current_role, "content": '\n'.join(current_content)})
        
        # Add the current user message
        messages.append({"role": "user", "content": user_message})
        
        # Define stop sequences
        stop_sequences = ["user:", "User:", "user :", "User :"]
        
        try:
            # Generate response using OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=1.0,
                top_p=0.95,
                max_tokens=100,
                frequency_penalty=0.2,  # Similar effect to repetition_penalty
                stop=stop_sequences
            )
            
            # Extract and return the generated text
            if response.choices and len(response.choices) > 0:
                text = response.choices[0].message.content
                if text:
                    return self.trim_to_last_sentence(text)
            return ""
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""
    
    def tokenize(self, text: str) -> List[int]:
        """Estimate tokenization (OpenAI doesn't expose tokenizer directly).
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            List[int]: Estimated list of token IDs (placeholder values)
        """
        # Rough estimation: ~4 characters per token for English text
        # This is a simplification; for accurate counts, use tiktoken
        try:
            import tiktoken
            # Try to get encoding for the model, fall back to cl100k_base
            try:
                encoding = tiktoken.encoding_for_model(self.model)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
            return encoding.encode(text)
        except ImportError:
            # Fallback: return placeholder token IDs based on character count
            estimated_tokens = len(text) // 4 + 1
            return list(range(estimated_tokens))
    
    def get_token_count(self, text: str) -> int:
        """Return token count of the input text.
        
        Args:
            text (str): Text to count tokens for
            
        Returns:
            int: Number of tokens
        """
        return len(self.tokenize(text))
    
    def batch_generate(self, prompts: List[Dict[str, str]], 
                       max_tokens: int = 512, 
                       temperature: float = 0.7) -> List[str]:
        """Generate responses for multiple prompts in a batch.        
        Args:
            prompts (List[Dict[str, str]]): List of prompt dictionaries, each with 
                                           'system', 'user' and optional 'history' keys
            max_tokens (int, optional): Maximum tokens to generate per response
            temperature (float, optional): Temperature for sampling
            
        Returns:
            List[str]: Generated responses
        """
        results = []
        stop_sequences = ["user:", "User:", "user :", "User :"]
        
        for p in prompts:
            system = p.get("system", "")
            user = p.get("user", "")
            history = p.get("history", "")
            
            # Build messages
            messages = [{"role": "system", "content": system}]
            
            # Parse history if provided
            if history:
                lines = history.strip().split('\n')
                current_role = None
                current_content = []
                
                for line in lines:
                    if line.startswith("User: "):
                        if current_role and current_content:
                            messages.append({"role": current_role, "content": '\n'.join(current_content)})
                        current_role = "user"
                        current_content = [line[6:]]
                    elif line.startswith("AI: "):
                        if current_role and current_content:
                            messages.append({"role": current_role, "content": '\n'.join(current_content)})
                        current_role = "assistant"
                        current_content = [line[4:]]
                    elif current_role:
                        current_content.append(line)
                
                if current_role and current_content:
                    messages.append({"role": current_role, "content": '\n'.join(current_content)})
            
            messages.append({"role": "user", "content": user})
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    top_p=0.95,
                    max_tokens=max_tokens,
                    frequency_penalty=0.2,
                    stop=stop_sequences
                )
                
                if response.choices and len(response.choices) > 0:
                    text = response.choices[0].message.content
                    results.append(text.strip() if text else "")
                else:
                    results.append("")
                    
            except Exception as e:
                print(f"Error in batch generation: {e}")
                results.append("")
                
        return results
