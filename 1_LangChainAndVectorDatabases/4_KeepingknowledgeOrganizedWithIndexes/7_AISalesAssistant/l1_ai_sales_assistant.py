text = """
Objection: "There's no money."
It could be that your prospect's business simply isn't big enough or generating enough cash right now to afford a product like yours. Track their growth and see how you can help your prospect get to a place where your offering would fit into their business.

Objection: "We don't have any budget left this year."
A variation of the "no money" objection, what your prospect's telling you here is that they're having cash flow issues. But if there's a pressing problem, it needs to get solved eventually. Either help your prospect secure a budget from executives to buy now or arrange a follow-up call for when they expect funding to return.

Objection: "We need to use that budget somewhere else."
Prospects sometimes try to earmark resources for other uses. It's your job to make your product/service a priority that deserves budget allocation now. Share case studies of similar companies that have saved money, increased efficiency, or had a massive ROI with you.
"""

# Split the text into a list using the keyword "Objection: "
objections_list = text.split("Objection: ")[1:]  # We ignore the first split as it is empty
# Now, prepend "Objection: " to each item as splitting removed it
objections_list = ["Objection: " + objection for objection in objections_list]
print(objections_list)

# We’re going to define a class that handles the database creation, database loading, and database querying.
import os
import re
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DeepLake

class DeepLakeLoader:
    def __init__(self, source_data_path):
        self.data = self.split_data()
        self.dataset_path=source_data_path
        self.deep_lake_token="eyJhbGciOiJIUzUxMiIsImlhdCI6MTcwMzk1NDQzMCwiZXhwIjoxNzM1NTc2ODE2fQ.eyJpZCI6InNyaXNodHlzdW1hbjI5MTkifQ.j_B5pt-7UL48fhp5cXStId-DNydAiuqDV3KlUC9cgE_ivDRyPaCmNnrWyTKp7Qkjx_rFL3M_T1yUR9HhG5vxmA"
        if False:
         self.db = self.load_db()
        else:
            self.db = self.create_db()


    def split_data(self):  
        """  
        Preprocess the data by splitting it into passages.  

        If using a different data source, this function will need to be modified.  

        Returns:  
            split_data (list): List of passages.  
        """  
        with open('/Users/srishtysuman/langchain_course/4_KeepingknowledgeOrganizedWithIndexes/7_AISalesAssistant/objections.txt', 'r') as f:  
            content = f.read()  
        split_data = re.split(r'(?=\d+\. )', content)
        if split_data[0] == '':  
            split_data.pop(0)  
        split_data = [entry for entry in split_data if len(entry) >= 30]  
        return split_data

    def load_db(self):  
        """  
        Load the database if it already exists.  

        Returns:  
            DeepLake: DeepLake object.  
        """  
        return DeepLake(token=self.deep_lake_token, dataset_path=self.dataset_path, embedding_function=OpenAIEmbeddings(openai_api_key="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"), read_only=True)  

    def create_db(self):  
        """  
        Create the database if it does not already exist.  

        Databases are stored in the deeplake directory.  

        Returns:  
            DeepLake: DeepLake object.  
        """  
        return DeepLake.from_texts(self.data, OpenAIEmbeddings(openai_api_key="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"), dataset_path=self.dataset_path, token=self.deep_lake_token)

    def query_db(self, query):  
        """  
        Query the database for passages that are similar to the query.  

        Args:  
            query (str): Query string.  

        Returns:  
            content (list): List of passages that are similar to the query.  
        """  
        results = self.db.similarity_search(query, k=3)  
        content = []  
        for result in results:  
            content.append(result.page_content)  
        return content

db = DeepLakeLoader('https:/app.activeloop.ai/srishtysuman2919/ai_sales_assistant')

results = db.query_db("We need to use that budget somewhere else.")


from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

chat = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="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")
system_message = SystemMessage(content="You are a Sales Call Assistant who detects potential objections from the customer (e.g. “It’s too expensive” or “The product doesn’t work for us”) and provide well-informed recommendations to the salesperson on how best to handle them.")
human_message = HumanMessage(content=f'Customer objection: {objections_list} | Relevant guidelines: {results}')

response = chat([system_message, human_message])

print(response.content)



