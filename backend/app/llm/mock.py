"""
PromptShield - Mock LLM Provider
Provides realistic responses without requiring an external API.
Used for development, testing, and demonstration.
"""

from typing import Optional
from app.llm.base import LLMProvider


class MockLLMProvider(LLMProvider):
    """
    Mock LLM provider that returns contextually appropriate responses
    without calling any external API.
    """
    
    RESPONSES = {
        "tcp": (
            "TCP (Transmission Control Protocol) establishes a connection using a "
            "three-way handshake:\n\n"
            "1. **SYN**: The client sends a SYN (synchronize) packet to the server, "
            "indicating it wants to establish a connection.\n\n"
            "2. **SYN-ACK**: The server responds with a SYN-ACK (synchronize-acknowledge) "
            "packet, confirming receipt and readiness.\n\n"
            "3. **ACK**: The client sends an ACK (acknowledge) packet back, completing "
            "the handshake.\n\n"
            "After this three-step process, a reliable, full-duplex communication "
            "channel is established between client and server."
        ),
        "python": (
            "Python is a high-level, interpreted programming language known for its "
            "readability and versatility. Key features include:\n\n"
            "- **Simple syntax** that emphasizes readability\n"
            "- **Dynamic typing** with strong type checking\n"
            "- **Extensive standard library** ('batteries included')\n"
            "- **Support for multiple paradigms**: OOP, functional, procedural\n"
            "- **Rich ecosystem** of third-party packages via PyPI\n\n"
            "Python is widely used in web development, data science, machine learning, "
            "automation, and scientific computing."
        ),
        "machine learning": (
            "Machine Learning (ML) is a subset of artificial intelligence where systems "
            "learn patterns from data without being explicitly programmed. Key concepts:\n\n"
            "- **Supervised Learning**: Training with labeled data (classification, regression)\n"
            "- **Unsupervised Learning**: Finding patterns in unlabeled data (clustering)\n"
            "- **Reinforcement Learning**: Learning through trial, error, and rewards\n\n"
            "Common algorithms include linear regression, decision trees, SVMs, neural networks, "
            "and ensemble methods like Random Forest and Gradient Boosting."
        ),
        "neural network": (
            "A neural network is a computational model inspired by the human brain, "
            "consisting of interconnected layers of nodes (neurons):\n\n"
            "- **Input Layer**: Receives the raw data\n"
            "- **Hidden Layers**: Process and transform the data through weighted connections\n"
            "- **Output Layer**: Produces the final prediction\n\n"
            "Each connection has a weight that's adjusted during training through "
            "backpropagation and gradient descent. Activation functions (ReLU, sigmoid, "
            "softmax) introduce non-linearity, enabling the network to learn complex patterns."
        ),
        "default": (
            "That's a great question! Based on my understanding, here's what I can share:\n\n"
            "This topic involves several important concepts that work together. "
            "The key principles include understanding the fundamental mechanisms, "
            "their interactions, and practical applications.\n\n"
            "For a more detailed explanation, I'd recommend exploring specific aspects "
            "of this topic. Feel free to ask follow-up questions about any particular "
            "area you'd like to dive deeper into."
        ),
        "encryption": (
            "Encryption is the process of converting plaintext into ciphertext to protect "
            "data confidentiality. The main types are:\n\n"
            "- **Symmetric Encryption** (AES, DES): Same key for encryption and decryption. "
            "Fast, used for bulk data.\n"
            "- **Asymmetric Encryption** (RSA, ECC): Public-private key pairs. Slower but "
            "enables secure key exchange.\n\n"
            "Modern systems often combine both: asymmetric encryption to exchange a symmetric "
            "key, then symmetric encryption for the actual data (hybrid encryption)."
        ),
        "api": (
            "An API (Application Programming Interface) is a set of protocols and tools "
            "that allows different software applications to communicate:\n\n"
            "- **REST APIs**: Use HTTP methods (GET, POST, PUT, DELETE) with JSON\n"
            "- **GraphQL**: Query language allowing clients to request specific data\n"
            "- **WebSocket**: Enables real-time, bidirectional communication\n"
            "- **gRPC**: High-performance RPC framework using Protocol Buffers\n\n"
            "APIs define endpoints, request/response formats, authentication methods, "
            "and rate limiting policies."
        ),
    }
    
    @property
    def name(self) -> str:
        return "MockLLM"
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response based on keyword matching."""
        prompt_lower = prompt.lower()
        
        for keyword, response in self.RESPONSES.items():
            if keyword == "default":
                continue
            if keyword in prompt_lower:
                return response
        
        return self.RESPONSES["default"]
    
    def is_available(self) -> bool:
        return True
