import os
import torch
import openai
from PIL import Image
from tqdm import tqdm
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.docstore.document import Document
from transformers import BlipProcessor, BlipForConditionalGeneration

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize BLIP model for image captioning
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)

# Set OpenAI API Key
openai.api_key = 'YOUR_OPENAI_API_KEY'  # Replace with your OpenAI API key

# LangChain Embedding Model (HuggingFace Embeddings)
embeddings_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

# Initialize Chroma client
chroma_client = chromadb.Client()

# Function to generate captions for meme images using BLIP model
def generate_image_caption(image_path: str):
    try:
        # Load and process the image
        image = Image.open(image_path).convert("RGB")
        inputs = blip_processor(images=image, return_tensors="pt").to(DEVICE)
        # Generate caption using BLIP model
        outputs = blip_model.generate(**inputs)
        caption = blip_processor.decode(outputs[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"Error generating caption for {image_path}: {e}")
        return "Unable to describe image"

# Function to create a prompt for LLM to analyze meme content
def generate_meme_analysis_prompt(caption: str):
    # You can improve this prompt to fit your exact needs
    prompt = f"""
    Based on the following meme caption, describe the meme's category and sentiment:
    Caption: {caption}
    Please provide the category (e.g., funny, sarcastic, motivational) and sentiment (positive, negative, neutral).
    """
    return prompt

# Function to classify meme using an LLM like OpenAI or another model
def classify_meme_with_gpt(caption: str):
    try:
        prompt = generate_meme_analysis_prompt(caption)
        
        response = openai.Completion.create(
            model="gpt-4",
            prompt=prompt,
            max_tokens=50,
            n=1,
            stop=None,
            temperature=0.7
        )
        
        result = response.choices[0].text.strip()
        return result
    except Exception as e:
        print(f"Error with GPT classification: {e}")
        return "Unknown"

# Function to load meme images and generate captions
def load_memes_and_generate_data(meme_dir: str):
    image_paths = []
    for root, _, files in os.walk(meme_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        raise FileNotFoundError(f"No images found in the directory: {meme_dir}")

    documents = []
    for image_path in tqdm(image_paths, desc="Processing images"):
        # Generate the caption for the meme image
        caption = generate_image_caption(image_path)
        # Classify the meme based on the caption
        meme_category = classify_meme_with_gpt(caption)
        
        # Store image features with embeddings using CLIP (optional)
        image = Image.open(image_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs).cpu().numpy().flatten()

        # Create a document with the image and metadata (caption, category)
        documents.append(
            Document(page_content=str(image_features), metadata={"image_path": image_path, "category": meme_category, "caption": caption})
        )

    return documents

# Function to create the Chroma index and store meme data
def create_chroma_index(documents):
    # Create a collection in Chroma
    chroma_collection = chroma_client.create_collection("meme_collection")
    
    # Create vector store in Chroma
    vector_store = Chroma.from_documents(
        documents=documents,
        embeddings=embeddings_model,
        collection_name="meme_collection"
    )
    
    return vector_store

# Function to search memes in Chroma based on a query
def search_memes(vector_store, query: str, top_k: int = 5):
    query_embedding = embeddings_model.embed_query(query)
    results = vector_store.similarity_search_with_score(query_embedding, k=top_k)
    
    retrieved_images = []
    for result in results:
        image_path = result[0].metadata['image_path']
        meme_category = result[0].metadata['category']
        caption = result[0].metadata['caption']
        retrieved_images.append((image_path, meme_category, caption))
    
    return retrieved_images

# Function to display image
def display_image(image_path: str):
    try:
        img = cv2.imread(image_path)
        if img is not None:
            # Display the image in an OpenCV window
            cv2.imshow(f"Image: {os.path.basename(image_path)}", img)
            cv2.waitKey(0)  # Wait for a key press
            cv2.destroyAllWindows()
        else:
            print(f"Error reading image: {image_path}")
    except Exception as e:
        print(f"Error displaying image {image_path}: {e}")

if __name__ == "__main__":
    meme_dir = "meme"  # Replace with your meme directory

    try:
        # Load memes and generate data for Chroma
        documents = load_memes_and_generate_data(meme_dir)
        vector_store = create_chroma_index(documents)

        while True:  # Loop for continuous queries
            query = input("Enter your search query (or type 'exit' to quit): ")
            if query.lower() == "exit":
                break

            retrieved_images = search_memes(vector_store, query)

            if retrieved_images:
                print(f"Top memes related to '{query}':")
                for image_path, meme_category, caption in retrieved_images:
                    print(f"Category: {meme_category}, Caption: {caption}, Image: {image_path}")
                    display_image(image_path)  # Display the retrieved image
            else:
                print("No matching memes found.")

    except FileNotFoundError as e:
        print(e)
        print("Make sure the 'memes' directory exists and contains image files.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
