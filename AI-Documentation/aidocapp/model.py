import llama_cpp
import copy


class Model:
    def __init__(
        self,
        local_model: "str | None" = None,
    ):
        if local_model is not None:
            try:
                self.llm = llama_cpp.Llama(
                    model_path=local_model,
                    n_gpu_layers=-1,
                    chat_format="llama-3",
                    verbose=False,
                )
            except Exception as e:
                print(f"Model loading error: {str(e)}")

        self.template = [{
            "role": "user",
            "content": (
                "Add a detailed docstring in the style of PEP 257 to the following "
                "{} method {}. The docstring should include: - A concise summary of "
                "the method's purpose.- A detailed description of each argument "
                "(name and type). - A description of the return value (if any). "
                " Add inline comments within the method body to explain complex "
                "logic or non-obvious steps. Return the method implementation with "
                "the docstring and inline comments as a single markdown code block. "
                "Do not modify the code. Do not add any chat-like comments."
            ),
        }]

    def generate_comments(self, code, language):

        prompt_copy = copy.deepcopy(self.template)
        prompt_copy[0]["content"] = prompt_copy[0]["content"].format(language, code)

        comment = self.llm.create_chat_completion(prompt_copy)["choices"][0]["message"]["content"]
        code_comment = comment[comment.find('\n') + 1:comment.rfind('\n')]

        return code_comment
