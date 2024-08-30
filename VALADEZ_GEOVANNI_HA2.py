# Natural Language Toolkit: Eliza
#
# Copyright (C) 2001-2023 NLTK Project
# Authors: Steven Bird <stevenbird1@gmail.com>
#          Edward Loper <edloper@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

# Based on an Eliza implementation by Joe Strout <joe@strout.net>,
# Jeff Epler <jepler@inetnebr.com> and Jez Higgins <mailto:jez@jezuk.co.uk>.

# a translation table used to convert things you say into things the
# computer says back, e.g. "I am" --> "you are"

from nltk.chat.util import Chat, reflections
from nltk.stem.porter import *
from nltk.stem import *
from nltk.tokenize import word_tokenize


# a table of response pairs, where each pair consists of a
# regular expression, and a list of possible responses,
# with group-macros labelled as %1, %2.

pairs = (
    (
        #School prompt (finals). (.*) means . as many times. Replies relate to user's feeling of finals. 
        #%1 is the token
        r"Finals (.*)",
        (
            "I hear you. Finals week %1 for many. Do you want to tell me more?",
            "I hear you when you say finals %1.",
            "Any idea as to why finals %1?",
        ),
    ),
    (
        #School prompt (homework). (.*) means . as many times. Replies relate to user's thoughts on homework
        #%1 is the token
        r"Homework (.*)",
        (
            "I see. Homework %1 for many.",
            "I hear you when you say homework %1.",
            "Any idea as to why homework %1?",
        ),
    ),
    (
        #School prompt (class). (.*) means . as many times. Replies relate to user's thougths on classes. 
        #%1 is the token
        r"Class (.*)", 
        (
            "I hear you. Class %1.",
            "I hear you when you say class %1.",
            "Any idea as to why class %1?",
        ),
    ),
    (
        #School prompt (commuting). (.*) means . as many times. Replies relate to user's thoughts on commuting. 
        #%1 is the token
        r"Commuting (.*)",
        (
            "Commuting %1. Definitely.",
            "I hear you when you say commuting %1.",
            "How do you feel when you say commuting %1?",
        ),
    ),
    (
        #School prompt (studying). (.*) means . as many times. Replies relate to thoughts on studying. 
        #%1 is the token
        r"Studying (.*)",
        (
            "Studying %1 for many. Do you want to tell me more?",
            "I agree when you say studying %1.",
            "How much are you studying to say studying %1?",
        ),
    ),
    (
        #Need prompt. (.*) means . as many times. Replies relate to user's need in token. %1 is the token
       
        r"I need (.*)",
        (
            "Why do you need %1?",
            "Would it really help you to get %1?",
            "Are you sure you need %1?",
        ),
    ),
    (
        #Don't prompt. (.*) means . as many times. Replies relate to user's assumption of ELIZA's inability in token.
        # %1 is the token
        r"Why don\'t you (.*)",
        (
            "Do you really think I don't %1?",
            "Perhaps eventually I will %1.",
            "Do you really want me to %1?",
        ),
    ),
    (
        #Can't prompt. (.*) means . as many times. Replies relate to user's assumption of their inability in token.
        # %1 is the token
        r"Why can\'t I (.*)",
        (
            "Do you think you should be able to %1?",
            "If you could %1, what would you do?",
            "I don't know -- why can't you %1?",
            "Have you really tried?",
        ),
    ),
    (
        #I can't prompt. (.*) means . as many times. Replies relate to user's assumption of their inability in token.
        # %1 is the token
        r"I can\'t (.*)",
        (
            "How do you know you can't %1?",
            "Perhaps you could %1 if you tried.",
            "What would it take for you to %1?",
        ),
    ),
    (
        #I am prompt. (.*) means . as many times. Replies relate to user input of their status quo. %1 is the token
        r"I am (.*)",
        (
            "Did you come to me because you are %1?",
            "How long have you been %1?",
            "How do you feel about being %1?",
        ),
    ),
    (
        #I'm prompt. (.*) means . as many times. Replies relate to user input of their status quo. %1 is the token
        r"I\'m (.*)",
        (
            "How does being %1 make you feel?",
            "Do you enjoy being %1?",
            "Why do you tell me you're %1?",
            "Why do you think you're %1?",
        ),
    ),
    (
        #Are you prompt. (.*) means . as many times. Replies relate assumption of what ELIZA is. %1 is the token
        r"Are you (.*)",
        (
            "Why does it matter whether I am %1?",
            "Would you prefer it if I were not %1?",
            "Perhaps you believe I am %1.",
            "I may be %1 -- what do you think?",
        ),
    ),
    (
        #What prompt. (.*) means . as many times. Replies relate to asking a question.
        r"What (.*)",
        (
            "Why do you ask?",
            "How would an answer to that help you?",
            "What do you think?",
        ),
    ),
    (
        #How prompt. (.*) means . as many times. Replies relate to reiiterating the how question. 
        r"How (.*)",
        (
            "How do you suppose?",
            "Perhaps you can answer your own question.",
            "What is it you're really asking?",
        ),
    ),
    (
        #Because prompt. (.*) means . as many times. %1 is the token
        r"Because (.*)",
        (
            "Is that the real reason?",
            "What other reasons come to mind?",
            "Does that reason apply to anything else?",
            "If %1, what else must be true?",
        ),
    ),
    (
        #Apoloigies prompt. (.*) means . as many times. Replies relate to aplogies from user.
        r"(.*) sorry (.*)",
        (
            "There are many times when no apology is needed.",
            "What feelings do you have when you apologize?",
        ),
    ),
    (
        #Greeting prompt. (.*) means . as many times. Replies are greetings. 
        r"Hello(.*)",
        (
            "Hello... I'm glad you could drop by today.",
            "Hi there... how are you today?",
            "Hello, how are you feeling today?",
        ),
    ),
    (
        r"I think (.*)",
        ("Do you doubt %1?", "Do you really think so?", "But you're not sure %1?"),
    ),
    ( 
        #Friend prompt. (.*) means . as many times. Replies relate to friends.
        r"(.*) friend (.*)",
        (
            "Tell me more about your friends.",
            "When you think of a friend, what comes to mind?",
            "Why don't you tell me about a childhood friend?",
        ),
    ),
    (r"Yes", ("You seem quite sure.", "OK, but can you elaborate a bit?")),
    (
        #Computer prompt. (.*) means . as many times. Replies relate to discussing computer.
        r"(.*) computer(.*)",
        (
            "Are you really talking about me?",
            "Does it seem strange to talk to a computer?",
            "How do computers make you feel?",
            "Do you feel threatened by computers?",
        ),
    ),
    (
        #It is prompt. (.*) means . as many times. Replies relate to assumed status quo. %1 is the token
        r"Is it (.*)",
        (
            "Do you think it is %1?",
            "Perhaps it's %1 -- what do you think?",
            "If it were %1, what would you do?",
            "It could well be that %1.",
        ),
    ),
    (
        #It is prompt. (.*) means . as many times. Replies relate to assumed status quo. %1 is the token 
        r"It is (.*)",
        (
            "You seem very certain.",
            "If I told you that it probably isn't %1, what would you feel?",
        ),
    ),
    (
        #Can you prompt. (.*) means . as many times. Replies relate to ELIZA ability to do token. %1 is the token 
        r"Can you (.*)",
        (
            "What makes you think I can't %1?",
            "If I could %1, then what?",
            "Why do you ask if I can %1?",
        ),
    ),
    (
        #Can I prompt. (.*) means . as many times. Replies relate to ability to do token. %1 is the token 
        r"Can I (.*)",
        (
            "Perhaps you don't want to %1.",
            "Do you want to be able to %1?",
            "If you could %1, would you?",
        ),
    ),
    (
        #You are prompt. (.*) means . as many times. Replies relate to Eliza chatbot. %1 is the token 
        r"You are (.*)",
        (
            "Why do you think I am %1?",
            "Does it please you to think that I'm %1?",
            "Perhaps you would like me to be %1.",
            "Perhaps you're really talking about yourself?",
        ),
    ),
    (
        #You're prompt. (.*) means . as many times. Replies relate to ELIZA chatbot. %1 is the token 
        r"You\'re (.*)",
        (
            "Why do you say I am %1?",
            "Why do you think I am %1?",
            "Are we talking about you, or me?",
        ),
    ),
    (
        #I don't prompt. (.*) means . as many times. Replies relate to not token. %1 is the token 
        r"I don\'t (.*)",
        ("Don't you really %1?", "Why don't you %1?", "Do you want to %1?"),
    ),
    (
        #I feel prompt. (.*) means . as many times. Replies relate to feeling token(state of being, good / bad).
        #%1 is the token 
        r"I feel (.*)",
        (
            "Good, tell me more about these feelings.",
            "Do you often feel %1?",
            "When do you usually feel %1?",
            "When you feel %1, what do you do?",
        ),
    ),
    (
        #I have prompt. (.*) means . as many times. Replies relate to having token. %1 is the token 
        r"I have (.*)",
        (
            "Why do you tell me that you've %1?",
            "Have you really %1?",
            "Now that you have %1, what will you do next?",
        ),
    ),
    (
        #I would prompt. (.*) means . as many times. Replies relate to would doing. Can be a question of why do token
        #%1 is the token 
        r"I would (.*)",
        (
            "Could you explain why you would %1?",
            "Why would you %1?",
            "Who else knows that you would %1?",
        ),
    ),
    (
        #Is there prompt. (.*) means . as many times. Replies relate to the existence of token. %1 is the token 
        r"Is there (.*)",
        (
            "Do you think there is %1?",
            "It's likely that there is %1.",
            "Would you like there to be %1?",
        ),
    ),
    (
        #My prompt. (.*) means . as many times. Replies relate to your object (token). %1 is the token 
        r"My (.*)",
        (
            "I see, your %1.",
            "Why do you say that your %1?",
            "When your %1, how do you feel?",
        ),
    ),
    (
        #You prompt. Replies relate to you. %1 is the token 
        r"You (.*)",
        (
            "We should be discussing you, not me.",
            "Why do you say that about me?",
            "Why do you care whether I %1?",
        ),
    ),
    (r"Why (.*)", ("Why don't you tell me the reason why %1?", "Why do you think %1?")),
    (
        #I want prompt. (.*) means . as many times. Replies relate to wanting. %1 is the token 
        r"I want (.*)",
        (
            "What would it mean to you if you got %1?",
            "Why do you want %1?",
            "What would you do if you got %1?",
            "If you got %1, then what would you do?",
        ),
    ),
    (
        #Mother keyword. Replies relate to mother.
        r"(.*) mother(.*)",
        (
            "Tell me more about your mother.",
            "What was your relationship with your mother like?",
            "How do you feel about your mother?",
            "How does this relate to your feelings today?",
            "Good family relations are important.",
        ),
    ),
    ( 
        #Father keyword. Replies relate to father.
        r"(.*) father(.*)",
        (
            "Tell me more about your father.",
            "How did your father make you feel?",
            "How do you feel about your father?",
            "Does your relationship with your father relate to your feelings today?",
            "Do you have trouble showing affection with your family?",
        ),
    ),
    ( 
        #Child keyword. (.*) means . as many times.  Replies relate to children.
        r"(.*) child(.*)",
        (
            "Did you have close friends as a child?",
            "What is your favorite childhood memory?",
            "Do you remember any dreams or nightmares from childhood?",
            "Did the other children sometimes tease you?",
            "How do you think your childhood experiences relate to your feelings today?",
        ),
    ),
    (
        #Question prompt. (.*) means . as many times.
        r"(.*)\?",
        (
            "Why do you ask that?",
            "Please consider whether you can answer your own question.",
            "Perhaps the answer lies within yourself?",
            "Why don't you tell me?",
        ),
    ),
    (
        #Quit prompt
        r"quit",
        (
            "Thank you for talking with me.",
            "Good-bye.",
            "Thank you, that will be $150.  Have a good day!",
        ),
    ),
    (
        #General response for when no input matches in our list.
        r"(.*)",
        (
            "Please tell me more.",
            "Let's change focus a bit... Tell me about your family.",
            "Can you elaborate on that?",
            "Why do you say that %1?",
            "I see.",
            "Very interesting.",
            "%1.",
            "I see.  And what does that tell you?",
            "How does that make you feel?",
            "How do you feel when you say that?",
        ),
    ),
)
#initializes chatbot
eliza_chatbot = Chat(pairs, reflections)


#eliza_chat drops initial comment
def eliza_chat():
    print("Therapist\n---------")
    print("Talk to the program by typing in plain English, using normal upper-")
    print('and lower-case letters and punctuation.  Enter "quit" when done.')
    print("=" * 72)
    print("Hello.  How are you feeling today?")
#eliza_chat drops conversational comment
stemmer = PorterStemmer()  # Initialize the stemmer here
while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Eliza: Goodbye!")
            break
        #tokenize input
        tokens = word_tokenize(user_input)
        #stems tokens
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        #returns stemmed response
        stemmed_input = " ".join(stemmed_tokens)
        response = eliza_chatbot.respond(stemmed_input)

        print(f"Eliza: {response}")
        eliza_chatbot.converse()
    



def demo():
    eliza_chat()



if __name__ == "__main__":
    demo()
    