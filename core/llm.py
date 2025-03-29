"""LLM interface for RAG responses."""
from typing import AsyncGenerator, List, Optional, Any
import structlog

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from config.config import LLMConfig

class LLMInterface:
    """Interface for language model interactions."""
    
    def __init__(self, settings: LLMConfig):
        """
        Initialize the LLM interface.
        
        Args:
            settings: Configuration settings for the language model
        """
        self.logger = structlog.get_logger().bind(component="LLMInterface")
        self.settings = settings
        
        # Initialize ChatOpenAI
        self.llm = ChatOpenAI(
            openai_api_key=settings.openai_api_key,
            base_url=settings.base_url,
            model_name=settings.model_name,
            temperature=settings.temperature,
            streaming=settings.streaming,
            max_tokens=settings.max_tokens,
            timeout=120,  # Increased timeout for reliability
            max_retries=3,
        )
        
        # Initialize memory for chat history
        self.messages = []
        self.memory_k = settings.memory_k if hasattr(settings, 'memory_k') else 25
        
        # Create system prompt
        system_prompt = """
        You are a helpful AI Assistant with document search and retrieval capabilities. Answer questions based on the provided context.
        Provide the detailed explanation.
        The provided context is a list of documents from a vector store knowledge base.
        The similarity score for each document is also provided as Euclidean distance where the lower the number the more similar.
        If the context doesn't contain relevant information, use your general knowledge but mention this fact. Keep answers focused and relevant to the query.
        If there is no context provided and you don't know, then answer "I don't know".
        """
        self.system_prompt = SystemMessage(content=system_prompt)
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                self.system_prompt,
                ("placeholder", "{context}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "Question: {input}"),
            ]
        )
        
        # Create the chain
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    async def generate_response(
        self, 
        user_input: str,
        context: str
    ) -> AsyncGenerator[str, None]:
        """
        Generate RAG-enhanced streaming response.
        
        Args:
            user_input: User's question
            context: Retrieved context from documents
            
        Yields:
            Response chunks
        """
        try:
            # Debug context information
            self.logger.debug(
                "Context information",
                context_length=len(context),
                context_snippet=context[:100] + "..." if len(context) > 100 else context,
                user_input=user_input
            )
            
            # Format context as a message
            context_msg = HumanMessage(content=f"<context>\n{context}\n</context>\n")
            
            # Get chat history
            history = self.get_chat_history()
            self.logger.debug(
                "Chat history information",
                history_length=len(history),
                messages_count=len(self.messages)
            )
            
            # Debug prompt information
            self.logger.debug(
                "Preparing to send to LLM",
                model=self.settings.model_name,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens
            )
            
            # Stream response
            response_content = ""
            async for chunk in self.chain.astream(
                {
                    "input": user_input,
                    "context": [context_msg],
                    "chat_history": history,
                }
            ):
                yield chunk
                response_content += chunk
            
            # Update memory
            self.messages.append(HumanMessage(content=user_input))
            self.messages.append(AIMessage(content=response_content))
            
        except Exception as e:
            self.logger.error(
                "Error generating response",
                error=str(e),
                user_input=user_input,
            )
            yield f"Error: {str(e)}"
    
    def get_chat_history(self) -> List[Any]:
        """
        Get the chat history for the context window.
        
        Returns:
            List of message objects
        """
        # Limit to the last k exchanges
        if len(self.messages) > self.memory_k * 2:
            return self.messages[-self.memory_k * 2:]
        return self.messages
