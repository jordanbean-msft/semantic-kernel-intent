from semantic_kernel.kernel_pydantic import KernelBaseModel
from semantic_kernel.plugin_definition import kernel_function, kernel_function_context_parameter

class HVACPlugin(KernelBaseModel):
    def __init__(self, kernel):
        self.kernel = kernel