import customtkinter as ctk
from tkinter import scrolledtext
import threading, queue, re, ast, operator
from transformers import pipeline
import warnings

# Suppress HuggingFace warnings for a cleaner terminal
warnings.filterwarnings("ignore")

# --- Global Style Configuration ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# ==========================================
# MODULE 1: Deterministic Logic Engine
# ==========================================
def safe_calculate(expr):
    """Safely evaluates mathematical expressions, bypassing the LLM."""
    expr = re.sub(r'[^0-9\+\-\*\/\(\)\.]', '', expr)
    if not expr: return None
    
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow}
    try:
        node = ast.parse(expr, mode='eval').body
        def _eval(node):
            if isinstance(node, ast.Constant): return node.value
            elif isinstance(node, ast.BinOp):
                return ops[type(node.op)](_eval(node.left), _eval(node.right))
            elif isinstance(node, ast.UnaryOp):
                return ops[type(node.op)](_eval(node.operand))
            else: raise TypeError(node)
        return str(_eval(node))
    except Exception:
        return None

# ==========================================
# MODULE 2: The Application Core
# ==========================================
class HarryGPT(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Harry AI — Advanced Agentic Assistant")
        self.geometry("1100x850")

        # Grid Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar ---
        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        
        self.logo = ctk.CTkLabel(self.sidebar, text="HARRY AI", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo.pack(pady=30)
        
        self.new_chat_btn = ctk.CTkButton(self.sidebar, text="Wipe Memory", command=self.clear_chat)
        self.new_chat_btn.pack(pady=10, padx=20)

        self.status_var = ctk.StringVar(value="Status: Initializing Core...")
        self.status_lbl = ctk.CTkLabel(self.sidebar, textvariable=self.status_var, font=("Inter", 11))
        self.status_lbl.pack(side="bottom", pady=20)

        # --- Chat Interface ---
        self.chat_display = scrolledtext.ScrolledText(self, state='disabled', wrap='word', 
                                                     bg="#1a1a1a", fg="#ececf1", font=("Inter", 12),
                                                     padx=25, pady=25, borderwidth=0)
        self.chat_display.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        # --- Input Console ---
        self.input_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.input_frame.grid(row=1, column=1, padx=20, pady=(0, 20), sticky="ew")
        
        self.entry = ctk.CTkTextbox(self.input_frame, height=100, font=("Inter", 12), corner_radius=15)
        self.entry.pack(side='left', fill='x', expand=True, padx=(0, 10))
        
        self.send_btn = ctk.CTkButton(self.input_frame, text="Execute", width=120, height=45, 
                                     command=self.on_send, corner_radius=8)
        self.send_btn.pack(side='right')

        # --- AI Brain State ---
        self.history = []
        self.model = None
        self.q = queue.Queue()
        self.after(100, self.process_queue)
        
        # Load the neural network in the background
        threading.Thread(target=self.load_engine, daemon=True).start()

    def load_engine(self):
        try:
            self.model = pipeline("text2text-generation", model="google/flan-t5-large")
            self.status_var.set("Status: Core Online")
        except Exception as e:
            self.status_var.set("Status: Load Error")
            print(f"Neural Error: {e}")

    def on_send(self):
        user_input = self.entry.get("1.0", "end").strip()
        if not user_input: return
        
        self.entry.delete("1.0", "end")
        self.update_display("YOU", user_input)
        
        self.status_var.set("Harry is thinking...")
        threading.Thread(target=self.think, args=(user_input,), daemon=True).start()

    def think(self, user_input):
        # 1. THE ROUTER: Math Interception
        math_match = re.search(r'\d+\s*[\+\-\*\/]\s*\d+', user_input)
        if math_match:
            calc_result = safe_calculate(math_match.group(0))
            if calc_result:
                self.finalize_response(user_input, f"The exact answer is {calc_result}.")
                return

        # 2. THE ROUTER: Loop Breakers
        lower_input = user_input.lower().strip()
        if lower_input in ["hello", "hi", "how are you", "how are you?"]:
            self.finalize_response(user_input, "My neural circuits are functioning perfectly. How can I assist you with your code today?")
            return

        # 3. LLM Processing
        if not self.model:
            self.finalize_response(user_input, "Error: Neural engine offline.")
            return

        system_prompt = "You are Harry, an expert technical assistant. Answer directly and never repeat the user.\n\n"
        
        # Build Context Window
        recent_history = self.history[-4:] 
        context_string = ""
        for turn in recent_history:
            context_string += f"{turn['speaker']}: {turn['text']}\n"
            
        full_prompt = f"{system_prompt}Context:\n{context_string}User: {user_input}\nHarry:"

        try:
            # Extreme penalty to stop echoing
            result = self.model(
                full_prompt,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=4.0 
            )
            response = result[0]['generated_text'].strip()
            
            # Clean output
            for tag in ["HARRY:", "Harry:", "Assistant:", "USER:"]:
                response = response.replace(tag, "")
            response = response.strip()
        except Exception as e:
            response = f"Inference Error: {str(e)}"

        self.finalize_response(user_input, response)

    def finalize_response(self, user_input, response_text):
        """Logs history and sends response to UI thread."""
        self.history.append({"speaker": "USER", "text": user_input})
        self.history.append({"speaker": "HARRY", "text": response_text})
        self.q.put(response_text)

    def update_display(self, speaker, text):
        self.chat_display.configure(state='normal')
        tag = speaker
        self.chat_display.insert('end', f"{speaker}\n", tag)
        self.chat_display.insert('end', f"{text}\n\n")
        
        # Styling for chat bubbles
        if speaker == "YOU":
            self.chat_display.tag_config(tag, foreground="#10a37f", font=("Inter", 11, "bold"))
        else:
            self.chat_display.tag_config(tag, foreground="#5dade2", font=("Inter", 11, "bold"))
            
        self.chat_display.see('end')
        self.chat_display.configure(state='disabled')

    def process_queue(self):
        try:
            while True:
                response = self.q.get_nowait()
                self.update_display("HARRY", response)
                self.status_var.set("Status: Core Online")
        except queue.Empty:
            pass
        self.after(100, self.process_queue)

    def clear_chat(self):
        self.chat_display.configure(state='normal')
        self.chat_display.delete("1.0", "end")
        self.chat_display.configure(state='disabled')
        self.history = []
        self.status_var.set("Status: Memory Wiped")

if __name__ == "__main__":
    app = HarryGPT()
    app.mainloop()