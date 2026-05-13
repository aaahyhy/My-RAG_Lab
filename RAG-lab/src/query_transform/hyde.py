class HyDE:

    def __init__(self, generator):

        self.generator = generator

    def transform(self, query):
        # prompt = f"""Please write a scientific abstract-style passage to answer the following research question.
        # Use professional terminology related to LLM safety and control science.
        #
        # Question: {query}
        # """
        prompt = f"""
        Please generate a passage that answers the following question.

        The passage should resemble a paragraph from a scientific paper.

        Question: {query}

        Passage:
        """

        return self.generator.generate(prompt)