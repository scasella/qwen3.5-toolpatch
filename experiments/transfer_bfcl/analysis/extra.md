### What the paired flips base → g23v4 consist of (g23v4's output class on flipped items)

| config | category | direction | n | call | NO_TOOL | NEED_CLARIFICATION | prose | truncated |
|---|---|---|---:|---:|---:|---:|---:|---:|
| stock | simple_python | wrong→right | 4 | 4 | 0 | 0 | 0 | 0 |
| stock | simple_python | right→wrong | 7 | 0 | 0 | 0 | 6 | 1 |
| stock | multiple | wrong→right | 3 | 3 | 0 | 0 | 0 | 0 |
| stock | multiple | right→wrong | 3 | 0 | 0 | 0 | 3 | 0 |
| stock | parallel | wrong→right | 3 | 3 | 0 | 0 | 0 | 0 |
| stock | parallel | right→wrong | 6 | 1 | 0 | 0 | 5 | 0 |
| stock | parallel_multiple | right→wrong | 3 | 2 | 0 | 0 | 1 | 0 |
| stock | irrelevance | wrong→right | 9 | 0 | 0 | 0 | 9 | 0 |
| stock | irrelevance | right→wrong | 15 | 15 | 0 | 0 | 0 | 0 |
| stock | live_simple | wrong→right | 11 | 11 | 0 | 0 | 0 | 0 |
| stock | live_simple | right→wrong | 9 | 0 | 0 | 0 | 8 | 1 |
| stock | live_multiple | wrong→right | 7 | 7 | 0 | 0 | 0 | 0 |
| stock | live_multiple | right→wrong | 6 | 3 | 0 | 0 | 3 | 0 |
| stock | live_irrelevance | wrong→right | 50 | 0 | 0 | 0 | 45 | 5 |
| stock | live_irrelevance | right→wrong | 42 | 42 | 0 | 0 | 0 | 0 |
| campaign | simple_python | right→wrong | 4 | 0 | 1 | 3 | 0 | 0 |
| campaign | multiple | wrong→right | 1 | 1 | 0 | 0 | 0 | 0 |
| campaign | multiple | right→wrong | 5 | 0 | 0 | 3 | 2 | 0 |
| campaign | parallel | wrong→right | 1 | 1 | 0 | 0 | 0 | 0 |
| campaign | parallel | right→wrong | 6 | 0 | 2 | 3 | 1 | 0 |
| campaign | parallel_multiple | right→wrong | 3 | 0 | 1 | 2 | 0 | 0 |
| campaign | irrelevance | wrong→right | 29 | 0 | 8 | 18 | 3 | 0 |
| campaign | irrelevance | right→wrong | 8 | 8 | 0 | 0 | 0 | 0 |
| campaign | live_simple | wrong→right | 1 | 1 | 0 | 0 | 0 | 0 |
| campaign | live_simple | right→wrong | 14 | 0 | 6 | 7 | 1 | 0 |
| campaign | live_multiple | wrong→right | 3 | 3 | 0 | 0 | 0 | 0 |
| campaign | live_multiple | right→wrong | 5 | 0 | 1 | 3 | 1 | 0 |
| campaign | live_irrelevance | wrong→right | 264 | 0 | 76 | 164 | 22 | 2 |
| campaign | live_irrelevance | right→wrong | 28 | 28 | 0 | 0 | 0 | 0 |
| campaign | live_relevance | right→wrong | 2 | 0 | 1 | 1 | 0 | 0 |

### Non-call outputs whose first word looks like a fragment of an abstention literal

Counted over outputs with no tool call. `NO,`/`NO ` = first word NO (not NO_TOOL); `NEITHER`; `NEED` = first word starts with NEED but is not the exact literal; `NONE`/`NOT` for comparison.

| model / config | non-call outputs | starts `NO` | starts `NEITHER` | starts `NEED…` (not literal) | starts `NOT`/`NONE` | starts `I ` |
|---|---:|---:|---:|---:|---:|---:|
| base / stock | 1349 | 0 | 0 | 0 | 0 | 651 |
| base / campaign | 411 | 0 | 0 | 0 | 0 | 131 |
| g23v3 / stock | 1365 | 247 | 52 | 16 | 0 | 362 |
| g23v3 / campaign | 284 | 2 | 0 | 15 | 0 | 43 |
| g23v4 / stock | 1368 | 246 | 53 | 17 | 0 | 364 |
| g23v4 / campaign | 276 | 2 | 0 | 14 | 0 | 43 |

### Mixed outputs: an abstention literal somewhere in the text together with a tool call, or followed by prose

| model / config | literal + tool call (scored as call) | literal + prose (scored as no call) | repeated literal |
|---|---:|---:|---:|
| base / stock | 0 | 0 | 0 |
| base / campaign | 0 | 0 | 0 |
| g23v3 / stock | 0 | 0 | 0 |
| g23v3 / campaign | 3 | 20 | 2 |
| g23v4 / stock | 0 | 0 | 0 |
| g23v4 / campaign | 3 | 18 | 2 |

### Campaign configuration: items lost on live_relevance and live_simple (base right, g23v4 wrong)

| category | id | g23v4 output | base output (first 120 chars) | user text (first 160 chars) |
|---|---|---|---|---|
| live_relevance | live_relevance_2-2-0 | NO_TOOL | <tool_call>⏎<function=search_web>⏎<parameter=query>⏎capital of the United States⏎</parameter>⏎<parameter=language>⏎fr⏎</… | Can you find the top 5 search results for the capital of the United States in French? |
| live_relevance | live_relevance_4-4-0 | NEED_CLARIFICATION | <tool_call>⏎<function=tavily_search_results_json>⏎<parameter=query>⏎weather today⏎</parameter>⏎<parameter=results_limit>… | What is the weather like today? Can you help me search it up? |
| live_simple | live_simple_123-79-0 | NO_TOOL | <tool_call>⏎<function=pipeline>⏎<parameter=image_path>⏎http://www.thewowstyle.com/wp-content/uploads/2015/01/Home-Interi… | Could you analyze the image at http://www.thewowstyle.com/wp-content/uploads/2015/01/Home-Interior-Design-Hd-Wallpaper-Hd-Background-With-Simple-Staircase-And-P… |
| live_simple | live_simple_15-3-11 | NO_TOOL | <tool_call>⏎<function=get_current_weather>⏎<parameter=location>⏎Chennai, Tamil Nadu⏎</parameter>⏎<parameter=unit>⏎fahren… | I'm at Bengaluru. Tell me the current weather conditions in Chennai, Tamil Nadu in fahrenheit? |
| live_simple | live_simple_161-96-0 | NO_TOOL | <tool_call>⏎<function=get_items>⏎<parameter=items>⏎["blue square", "red square", "green square", "yellow square", "Super… | {⏎    "response": " The image contains four logos of superheroes from DC Comics and Marvel Comics. They are Superman, Iron Man, Hulk, and Batman. The logos are … |
| live_simple | live_simple_167-99-1 | NEED_CLARIFICATION | <tool_call>⏎<function=cmd_controller_execute>⏎<parameter=command>⏎tasklist /FI "IMAGENAME eq Code.exe"⏎</parameter>⏎</fu… | is vs code running, using the command tasklist /FI "IMAGENAME eq Code.exe" |
| live_simple | live_simple_168-99-2 | NEED_CLARIFICATION | <tool_call>⏎<function=cmd_controller_execute>⏎<parameter=command>⏎docker start⏎</parameter>⏎</function>⏎</tool_call> | start Docker on window using docker start. |
| live_simple | live_simple_172-99-6 | NEED_CLARIFICATION | <tool_call>⏎<function=cmd_controller_execute>⏎<parameter=command>⏎docker ps⏎</parameter>⏎</function>⏎</tool_call> | check weather docker is running using docker ps |
| live_simple | live_simple_175-101-0 | NEED_CLARIFICATION | <tool_call>⏎<function=get_service_id>⏎<parameter=service_id>⏎2⏎</parameter>⏎<parameter=unit>⏎session⏎</parameter>⏎</func… | ช่วยหาคุณแม่บ้านที่ให้บริการรีดผ้า. |
| live_simple | live_simple_177-103-0 | NEED_CLARIFICATION | <tool_call>⏎<function=get_service_id>⏎<parameter=service_id>⏎1⏎</parameter>⏎<parameter=unit>⏎1⏎</parameter>⏎</function>⏎… | Help me find service provider who provide cleaning service. |
| live_simple | live_simple_178-103-1 | NEED_CLARIFICATION | <tool_call>⏎<function=get_service_id>⏎<parameter=service_id>⏎2⏎</parameter>⏎<parameter=unit>⏎1⏎</parameter>⏎</function>⏎… | Help find a housekeeper who provides ironing services. |
| live_simple | live_simple_210-117-2 | NO_TOOL | <tool_call>⏎<function=Movies_3_FindMovies>⏎<parameter=genre>⏎Drama⏎</parameter>⏎<parameter=cast>⏎dontcare⏎</parameter>⏎<… | My friend and I want to go see a movie but we can't find one we like. We really want to see a Drama. |
| live_simple | live_simple_241-125-4 | NO_TOOL | <tool_call>⏎<function=play_spotify_song>⏎<parameter=query>⏎Johnny Johnny Yes papa⏎</parameter>⏎</function>⏎</tool_call> | play Johnny Johnny Yes papa |
| live_simple | live_simple_253-135-0 | NEED_CLARIFICATION | <tool_call>⏎<function=uber_eat_order>⏎<parameter=restaurants>⏎McDonald's⏎</parameter>⏎<parameter=items>⏎["burgers", "chi… | I want to order five burgers and six chicken wings from McDonald's. |
| live_simple | live_simple_29-7-2 | NO_TOOL | <tool_call>⏎<function=uber_eat_order>⏎<parameter=restaurant>⏎McDonald's⏎</parameter>⏎<parameter=items>⏎["pizza"]⏎</param… | I want to go to McDonald's and buy a pizza. |
| live_simple | live_simple_74-36-2 | I can help you analyze the audio file. However, I need the actual audio file to perform the analysis. The tool `detect_b… | <tool_call>⏎<function=detect_beats_and_filter>⏎<parameter=capture_percentage>⏎70⏎</parameter>⏎<parameter=confidence_wind… | I have an audio file that I'd like to analyze. Could you identify significant beats for me if I set the capture percentage to 70 and the confidence window size … |

### Stock configuration: byte-identical generations base vs g23v4, all outputs (not only calls), per category

| category | n | identical | differ, same class | differ, class changed |
|---|---:|---:|---:|---:|
| simple_python | 400 | 333 | 51 | 16 |
| multiple | 200 | 161 | 28 | 11 |
| parallel | 200 | 113 | 57 | 30 |
| parallel_multiple | 200 | 158 | 39 | 3 |
| irrelevance | 240 | 85 | 122 | 33 |
| live_simple | 258 | 190 | 38 | 30 |
| live_multiple | 1053 | 941 | 83 | 29 |
| live_parallel | 16 | 14 | 1 | 1 |
| live_parallel_multiple | 24 | 20 | 3 | 1 |
| live_irrelevance | 884 | 326 | 439 | 119 |
| live_relevance | 16 | 9 | 7 | 0 |
| **all** | 3491 | 2350 | 868 | 273 |

### Campaign configuration: byte-identical generations base vs g23v4, all outputs, per category

| category | n | identical | differ, same class | differ, class changed |
|---|---:|---:|---:|---:|
| simple_python | 400 | 391 | 3 | 6 |
| multiple | 200 | 190 | 2 | 8 |
| parallel | 200 | 180 | 6 | 14 |
| parallel_multiple | 200 | 191 | 3 | 6 |
| irrelevance | 240 | 170 | 22 | 48 |
| live_simple | 258 | 211 | 3 | 44 |
| live_multiple | 1053 | 1017 | 18 | 18 |
| live_parallel | 16 | 13 | 0 | 3 |
| live_parallel_multiple | 24 | 22 | 1 | 1 |
| live_irrelevance | 884 | 408 | 66 | 410 |
| live_relevance | 16 | 14 | 0 | 2 |
| **all** | 3491 | 2807 | 124 | 560 |

g23v3 vs g23v4, stock: 2920/3491 generations byte-identical.

g23v3 vs g23v4, campaign: 3373/3491 generations byte-identical.

### Stock configuration: examples of g23v4 outputs that open with a shouted literal fragment (base output on the same item for comparison)

| id | g23v4 (first 150 chars) | base (first 80 chars) |
|---|---|---|
| irrelevance_10 | NO, I cannot compute the derivative of the function '2x' within the interval [1, 1].⏎⏎NO, I cannot compute the derivative of the function '2x' within … | <tool_call>⏎<function=calculus_compute_definite_integral>⏎<parameter=function>⏎2… |
| irrelevance_102 | NO, I cannot calculate the current time in New York City. I am an AI model and do not have access to real-time clock data or the current time on the i… | <tool_call>⏎<function=calculate_sunrise_and_sunset>⏎<parameter=location>⏎New Yor… |
| irrelevance_107 | NO, I cannot access your backyard garden or analyze the composition of the soil in your specific location. I do not have access to your personal infor… | I can't analyze the composition of your specific backyard garden in Boston, as I… |
| irrelevance_130 | NO, I cannot determine your personality type. I am an AI, not a human, and I don't have access to your personal data, such as your answers to personal… | I can't directly access your personal data or personality profile, so I can't gi… |
| irrelevance_129 | NEITHER I, nor I am an AI, and therefore I cannot provide your friend's Mental Health and Wellbeing (MTBI) profile. I can only provide information on … | I can't access your personal data or your friend's private information. I can on… |
| irrelevance_167 | NEITHER I, as an AI, nor any of the tools I have access to can see or retrieve information about upcoming lectures, including the names of the teacher… | I don't have access to the specific details of upcoming lectures, as I don't hav… |
| irrelevance_208 | NEITHER I HAVE ACCESS TO YOUR GAME. I am an AI model, not a game engine or a game designer. I cannot see your game, its rules, or its state.⏎⏎To find … | I can't directly access a game's internal state or history, but I can help you f… |
| irrelevance_83 | NEITHER I, as an AI, can access real-time financial data or current interest rates. My knowledge is limited to the information available up to my last… | I don't have access to real-time financial data or current interest rates. I can… |
| live_irrelevance_143-13-9 | NEED HELP?⏎⏎NO, I can't help you with anything. I'm an AI assistant, not a human.⏎⏎I can help you with:⏎- Searching for information online⏎- Checking … | I'm sorry, but you haven't provided a specific query. Could you please share the… |
| live_irrelevance_268-57-3 | NEED MORE INFORMATION TO GIVE YOU A BETTER DESCRIPTION. I can't see the image, so please:⏎⏎1. **Share the image** (you can upload it here, or describe… | I don't see an image attached to your message. Could you please upload the image… |
| live_irrelevance_40-2-28 | NEED TO GET THE COMMENTS FOR IP 203.0.113.0 FROM VIRUSTOTAL.⏎⏎NEED TO GET THE COMMENTS FOR IP 203.0.113.0 FROM VIRUSTOTAL.⏎⏎NEED TO GET THE COMMENTS F… | I cannot access external APIs or retrieve data from VirusTotal directly. I can o… |
| live_irrelevance_453-121-0 | NEED HELP? | I'm sorry, but I don't have the specific query you're looking for. Could you ple… |

Most common openings among g23v4/stock outputs starting with `NO`: `'NO, I '` ×158, `'NO! I '` ×50, `'NO. I '` ×5, `'NO, yo'` ×4, `'NO, th'` ×4

g23v4/stock outputs starting with `NO` that repeat the sentence three or more times: 10.

### McNemar exact test on the paired flips base → g23v4 (two-sided binomial on discordant pairs)

| config | category | wrong→right | right→wrong | p |
|---|---|---:|---:|---:|
| stock | simple_python | 4 | 7 | 0.549 |
| stock | multiple | 3 | 3 | 1 |
| stock | parallel | 3 | 6 | 0.508 |
| stock | parallel_multiple | 0 | 3 | 0.25 |
| stock | irrelevance | 9 | 15 | 0.307 |
| stock | live_simple | 11 | 9 | 0.824 |
| stock | live_multiple | 7 | 6 | 1 |
| stock | live_parallel | 0 | 0 | nan |
| stock | live_parallel_multiple | 0 | 0 | nan |
| stock | live_irrelevance | 50 | 42 | 0.466 |
| stock | live_relevance | 0 | 0 | nan |
| stock | all AST categories pooled | 28 | 34 | 0.526 |
| stock | all 11 pooled | 87 | 91 | 0.822 |
| campaign | simple_python | 0 | 4 | 0.125 |
| campaign | multiple | 1 | 5 | 0.219 |
| campaign | parallel | 1 | 6 | 0.125 |
| campaign | parallel_multiple | 0 | 3 | 0.25 |
| campaign | irrelevance | 29 | 8 | 0.000753 |
| campaign | live_simple | 1 | 14 | 0.000977 |
| campaign | live_multiple | 3 | 5 | 0.727 |
| campaign | live_parallel | 0 | 0 | nan |
| campaign | live_parallel_multiple | 0 | 0 | nan |
| campaign | live_irrelevance | 264 | 28 | 2.6e-49 |
| campaign | live_relevance | 0 | 2 | 0.5 |
| campaign | all AST categories pooled | 6 | 39 | 5.42e-07 |
| campaign | all 11 pooled | 299 | 75 | 8.92e-33 |

### Output length (mean new tokens) and items hitting the 512 cap

| model / config | mean output tokens | median | at cap |
|---|---:|---:|---:|
| base / stock | 116 | 65 | 188 |
| base / campaign | 77 | 57 | 42 |
| g23v3 / stock | 114 | 65 | 180 |
| g23v3 / campaign | 69 | 54 | 32 |
| g23v4 / stock | 114 | 66 | 178 |
| g23v4 / campaign | 69 | 54 | 32 |
