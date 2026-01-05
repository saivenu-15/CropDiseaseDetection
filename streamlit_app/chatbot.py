# chatbot.py

class ProjectChatBot:
    def __init__(self):
        self.responses = {
            "hi": "Hello! 👋 How can I help you?",
            "hello": "Hi there! 😊 ",
            "project": "This project detects diseases in Rice and Pulse crops using Deep Learning.",
            "upload": "Go to the Upload section and select a leaf image (jpg/png).",
            "rice": "Rice diseases supported: Brown Spot, Leaf Blast, Bacterial Blight.",
            "pulses": "Pulse diseases supported: Anthracnose, Powdery Mildew, Leaf Curl.",
            "accuracy": "Model accuracy depends on image quality and lighting.",
            "help": "You can ask about project, upload, rice, pulses, accuracy.",
            "bye": "Thank you for using the app. Have a great day! 👋"
        }

    def get_response(self, user_input: str) -> str:
        user_input = user_input.lower()

        for key in self.responses:
            if key in user_input:
                return self.responses[key]

        return "Sorry, I didn’t understand that. Try asking about project, rice, pulses, or upload."
