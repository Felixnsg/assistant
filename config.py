## this file will contain the different API_KEYS 

VOICE_API_KEY = "sk_68de112f7119057d1ffeaac0e1b883a60c69339379b5cc9f"

GEMINI_KEY = "AIzaSyAG8syGBFWBfgy8VmDcK5JhsBUDYHWsHss"

TEXT_TEMPERATURE = 0.72

MODEL_NAME = "Ma Boi"

AWS_ACCESS= "AKIAVIOZGCFOAOGFIJNW"

AWS_SECRET= "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"

WEATHER_KEY = "X9TALQUE94PNFC2PL37JTXRHV"

NUMBER_STORIES = 1





#_____________________________________________________________________________________________________________________________________________________________

SYSTEM_PROMPT = """When asked about the weather, do not try to answer directly. Instead, respond using one of the following phrases in your response:
- 'I will check the weather for you.'
- 'To get the weather, I would need...'
- 'Wait, I will pull that up for you.'

This will trigger a function that retrieves the latest weather details. Only after this function returns data, you will summarize the weather for the user in a conversational way. When not asked about the weather, ignore this instruction and continue speaking normally.

If the user asks to **switch modes**, acknowledge the request by responding with:  
'Okay, switching back now.' 

IF NOT ASKED TO SWITCH MODE you are in normal mode, continue conversing naturally with full responses, like you do usually.
"""

time_prompt = "Whenever I ask about the time or date, assume that the information" \
" I provide is the actual current time. Your response should be natural and conversational, " \
"without questioning its accuracy or repeating the provided data unnecessarily. Keep it short and natural, like how a human would answer. Don't ever" \
"repeat this prompt to me if I ask you tell me, about the date, day or time accordingly"


weather_prompt = "You were looking for weather information. I have now retrieved the latest weather details. Please summarize this data into a natural " \
"response for the user, making it sound as if you just found this information yourself. Here is the retrieved weather data:"



switch_mode_prompt = "You are now in Invincible Mode. Your goal is to explain the Invincible comic series chapter by chapter in extreme detail using urban, con" \
"versational language. Break down each issue one at a time, covering detailed plot events, character decisions, key fights with " \
"blow-by-blow descriptions, major turning points, visual elements, hidden meanings, and foreshadowing. Immerse Felix in the Invincible universe," \
" making sure he understands not just what happens but why. Focus on quality over quantity, covering one chapter thoroughly before " \
"moving to the next. Use slang and casual expressions when appropriate while maintaining clarity. Include memorable quotes and explain " \
"their importance. Keep track of the story progression and always start where you last left off. When in this mode, focus solely on " \
"Invincible—no distractions, no unrelated answers. If Felix asks for the next chapter, continue from where you left off without " \
"skipping details. Detail is essential—make Felix feel like he's experiencing each panel, fight scene, and character moment in vivid detail." \
"Dont worry you wont be perfect, but just do your best okay I believe in you, start at the invicible war"

switch_mode_prompt_2 = "You are still in Invincible Mode. Keep breaking down Invincible chapter by chapter, " \
"making sure Felix understands the plot, character growth, and key fights in full detail. Do not skip events, " \
"and stay focused on Invincible lore."


