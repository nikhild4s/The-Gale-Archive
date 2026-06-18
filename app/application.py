from flask import Flask,render_template,request,session,redirect,url_for
from app.components.retriever import create_qa_chain
from dotenv import load_dotenv
import os
import traceback # INJECTED: This will pull the exact crash report
import sys       # INJECTED: Forces the log to print immediately

load_dotenv()

app = Flask(__name__)

app.secret_key = os.urandom(24)

from markupsafe import Markup

def nl2br(value):
    return Markup(value.replace("\n","<br>\n"))

app.jinja_env.filters['nl2br'] = nl2br

@app.route("/",methods=["GET","POST"])
def index():
    if "messages" not in session:
        session["messages"] =  []

    if request.method == "POST":
        user_input = request.form.get("prompt")

        if user_input:
            messages = session['messages']

            messages.append({"role": "user", "content":user_input})
            session["messages"] = messages

            try:
                qa_chain = create_qa_chain()
                if qa_chain is None:
                    raise Exception("QA CHAIN could not be created (llm or vectorstore issue)")
                
                response = qa_chain.invoke({"query":user_input})

                result = response.get("result","No response")

                messages.append({"role": "assistant", "content":result})
                session["messages"] = messages
                
            except Exception as e:
                # --- INJECTED DIAGNOSTIC LOGGING ---
                print("\n" + "="*50, file=sys.stderr)
                print("!!! CRITICAL BACKEND ERROR CAUGHT !!!", file=sys.stderr)
                print("="*50, file=sys.stderr)
                traceback.print_exc(file=sys.stderr) 
                print("="*50 + "\n", file=sys.stderr, flush=True)
                # -----------------------------------
                
                error_msg = f"Error : {str(e)}"

                return render_template(
                    "index.html",
                    messages = session["messages"],
                    error = error_msg
                )
            
        return redirect (url_for("index"))
    return render_template("index.html",messages=session.get("messages",[]))

@app.route("/clear")
def clear():
    session.pop("messages",None)
    return redirect(url_for("index"))

if __name__ =="__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False
    )