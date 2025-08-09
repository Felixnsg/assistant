"""
Service dispatcher stage for executing utility services.

Handles detection and execution of service triggers in responses.
"""

import logging
from typing import Any, Optional

from ..base import PipelineStage
from ..context import PipelineContext


class ServiceDispatcher(PipelineStage):
    """
    Dispatches and executes utility services based on triggers.
    
    This stage is responsible for:
    - Detecting service triggers in LLM responses
    - Executing appropriate services
    - Handling visual context updates
    - Recording service results
    """
    
    def __init__(self, utilities_instance: Optional[Any] = None):
        """
        Initialize the service dispatcher.
        
        Args:
            utilities_instance: Optional utilities instance for service execution
        """
        super().__init__("ServiceDispatcher")
        self.utilities = utilities_instance
    
    async def process(self, context: PipelineContext) -> PipelineContext:
        """
        Check for and execute service triggers.
        
        Args:
            context: Current pipeline context
            
        Returns:
            Context with service results and possible visual context update
        """
        if not self.utilities:
            self.logger.debug("No utilities instance available, skipping service dispatch")
            return context
        
        if not context.llm_response:
            self.logger.debug("No LLM response to check for services")
            return context
        
        if not hasattr(self.utilities, 'dispatch_service'):
            self.logger.error("Utilities instance missing dispatch_service method")
            return context
        
        try:
            # Dispatch service if triggered
            service_result = await self.utilities.dispatch_service(context.llm_response)
            
            if service_result:
                service_name = service_result.get("service")
                result = service_result.get("result")
                
                self.logger.info(f"Service '{service_name}' executed for {context.request_id}")
                self.logger.debug(f"Service result: {str(result)[:200]}")
                
                # Handle visual context service specially
                if service_name == "CHECK_VISUAL_CONTEXT" and isinstance(result, dict):
                    visual_context = result.get("context_string")
                    if visual_context:
                        self.logger.info(f"Visual context updated: {visual_context[:100]}...")
                        # Store context for next turn
                        return context.with_update(
                            visual_context=visual_context,
                            service_results={service_name: result},
                            detected_services=[service_name]
                        )
                
                # Regular service result
                return context.with_update(
                    service_results={service_name: result},
                    detected_services=[service_name]
                )
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error during service dispatch: {e}", exc_info=True)
            # Don't fail the pipeline for service errors
            return context.with_update(
                errors=context.errors + [{
                    "stage": "ServiceDispatcher",
                    "error": str(e),
                    "severity": "warning"
                }]
            )