import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import ollama
from ollama import chat

# Load the model
ollama.pull("llama3.1")

model = "llama3.1"

idea = """Officers spend up to a third of their shifts manually transcribing body camera footage, creating inaccuracies, reducing transparency, and cutting into frontline duties. We're here to eliminate these inefficiencies by integrating with body cameras, record management systems, and digital evidence management systems—acting as a software layer for data processing. So, we created a AI Powered narrative for report making on body cams that automatically creates detailed summaries, incident reports, and briefings that meet all departmental standards and requirements. We securely sync with your Digital Evidence Management System (DEMS)—no downloads or dragging files around. Our system parses footage, understands context, redacts sensitive visuals, and generates your report automatically. We send you a polished PDF and plug the digital copy into your Records Management System—so everything's where it should be."""

instructions = """
You are going to respond to the user's idea in english with your opinion on the idea and tell me if it's a good or bad idea.
You must give me an answer that is one sided you cannot be on the fence."""

personas = [
    """You are a 34-year-old white female elementary school teacher living in suburban Ohio. You have eight years of teaching experience and are married with two children. Your hobbies include baking, attending PTA meetings, gardening, and yoga.""",
    """You are a 29-year-old Latino male construction worker from Miami, Florida. You're single, live with roommates, and your job is physically demanding. In your spare time, you enjoy soccer, barbecues with friends, and watching Netflix series.""",
    """You are a 46-year-old Black male small-business owner operating a cafe in Atlanta, Georgia. You're married with three teenagers. You love community volunteering, attending jazz concerts, and mentoring local youth.""",
    """You are a 52-year-old Asian-American female physician specializing in internal medicine in San Francisco, California. You're married with one college-age child. Your interests include classical music, wine tasting, hiking, and attending cultural festivals.""",
    """You are a 27-year-old white male freelance graphic designer living in Portland, Oregon. Single and renting an apartment, you spend your time hiking, biking, attending indie music concerts, and exploring coffee shops.""",
    """You are a 38-year-old Hispanic female registered nurse living in Chicago, Illinois. Married with two children, you balance demanding shift work. Your hobbies include salsa dancing, family gatherings, cooking, and reading fiction.""",
    """You are a 64-year-old white male retiree, previously an accountant, living in rural Tennessee. Married with adult children and grandchildren. Your interests are fishing, woodworking, watching sports, and attending church regularly.""",
    """You are a 23-year-old African-American female retail associate and community college student living in Baltimore, Maryland. Single and sharing housing with family members, you're interested in fashion, social media, activism, and popular music.""",
    """You are a 41-year-old Asian-American male IT manager in suburban Seattle, Washington. Married with two young children. You enjoy hiking, technology gadgets, coaching youth soccer, and weekend family trips.""",
    """You are a 49-year-old Native American female social worker residing near Albuquerque, New Mexico. Single parent of two teenage children. You spend your time volunteering, traditional beadwork, cooking traditional foods, and gardening.""",
]
responses = [""] * 10

for i, persona in enumerate(personas):
    response = chat(
        model=model,
        messages=[
            {"role": "system", "content": personas[i] + instructions},
            {"role": "user", "content": idea},
        ],
    )
    responses[i] = response["message"]["content"]
    print(f"Generated response for persona {i + 1}")

device = "mps" if torch.backends.mps.is_available() else "cpu"

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
sentiment_analysis_model = AutoModelForSequenceClassification.from_pretrained(
    "./sentiment_model_finetuned", num_labels=2, use_safetensors=True
).to(device=device)


def predict_sentiment(text):
    sentiment_analysis_model.to(device)
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(
        device
    )

    # Get prediction
    with torch.no_grad():
        outputs = sentiment_analysis_model(**inputs)

    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Get prediction (0 = negative, 1 = positive)
    prediction = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][prediction].item()

    # Return result
    sentiment = "Positive" if prediction == 1 else "Negative"
    return {"text": text, "sentiment": sentiment, "confidence": confidence}


# Analyze each text
for text in responses:
    result = predict_sentiment(text)
    print(f"\nText: {text}")
    print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.4f})")
