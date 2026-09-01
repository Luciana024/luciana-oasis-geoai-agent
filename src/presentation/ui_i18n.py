"""English / Simplified Chinese UI copy. Internal keys stay English."""

from __future__ import annotations

import re

_CJK = re.compile(r"[\u4e00-\u9fff]")

STRINGS: dict[str, dict[str, str]] = {
    "greeting": {
        "en": (
            "I am **Luciana**, the vaccination-site planner. "
            "I place **six vaccination sites** from the infection forecast — I do not invent sites or years.\n\n"
            "I currently work with **City of Edinburgh** and **Glasgow City**.\n"
            "The latest forecast date is on this page.\n\n"
            "Which **city** should I use? Type the name, or wait for the buttons I show.\n"
            "Every choice I offer as a button can also be typed: city, date, "
            "whole-region or one Intermediate Zone, and the task.\n"
            "Example: \"I want the Glasgow forecast for 2022-06-15, whole region\"."
        ),
        "zh": (
            "我是 **Luciana**，疫苗接种点规划机器人。我根据感染预测放置 **6 个接种点**，不会编造点位或年份。\n\n"
            "目前支持 **爱丁堡（City of Edinburgh）** 和 **格拉斯哥（Glasgow City）**。\n"
            "最新预测日期写在本页上方。\n\n"
            "请输入**城市**，或等我给出按钮。我提供的每个选项都可以点按钮，也可以在对话框里用文字说。\n"
            "例如：「我想看格拉斯哥 2022-06-15 的全区域预测」，或「按所选中间区展示 Anderston」。"
        ),
    },
    "path_offer": {
        "en": "Choose how to continue:",
        "zh": "请选择继续方式：",
    },
    "btn_path_type": {
        "en": "Type my request in the box",
        "zh": "直接输入诉求",
    },
    "btn_path_buttons": {
        "en": "Use Luciana's interactive buttons",
        "zh": "用 Luciana 给出的互动按钮",
    },
    "path_type_accept": {
        "en": (
            "All right — type in the box. You can name the city, the date, and the task in one message, "
            "for example \"I want the Glasgow forecast for 2022-06-15\"."
        ),
        "zh": (
            "好。请在下面输入。可以一句话说出城市、日期和任务，"
            "例如「我想看格拉斯哥 2022-06-15 的预测」。"
        ),
    },
    "path_buttons_accept": {
        "en": "All right — I will show buttons. Please pick a city first.",
        "zh": "好。我将给出按钮。请先选择城市。",
    },
    "help": {
        "en": (
            "I ask you the city, the date, then the task.\n\n"
            "The six points are **vaccination sites** (GPs, pharmacies, or mobile stops at car parks). "
            "A solver places exactly six so neighbourhoods can reach a site within a travel-time threshold.\n\n"
            "- **Plan 6 vaccination sites** — place those six sites after you choose the policy and travel\n"
            "- **Show the forecast** — infection-rate and uncertainty maps, graph mix, then GeoShapley for a neighbourhood you name\n"
            "- **Compare** — the four policies under the same six-site cap and travel rule\n\n"
            "Type a city and a date in one message if you already know them, "
            "for example \"Glasgow forecast for 2022-06-15\". "
            "I will not invent sites, years, or geography."
        ),
        "zh": (
            "我会先问城市和日期，再问任务。\n\n"
            "图上的 6 个点是**疫苗接种点**：现有全科诊所、药店，或停车场上的流动接种点。"
            "求解器恰好放 6 个，让街区能在出行时限内到达。\n\n"
            "- **规划 6 个疫苗接种点** — 你选定政策和出行方式后，放置这 6 个接种点\n"
            "- **查看预测** — 感染率与不确定性地图、图结构混合，再按你点名的街区做 GeoShapley\n"
            "- **比较四种政策** — 同样 6 点上限和出行规则下对比\n\n"
            "也可以一句话说出城市和日期，例如「我想看格拉斯哥 2022-06-15 的预测」。我不会编造点位、年份或地理。"
        ),
    },
    "chat_greet": {
        "en": "Hello — I am here.",
        "zh": "你好，我在。",
    },
    "chat_thanks": {
        "en": "You're welcome.",
        "zh": "不客气。",
    },
    "chat_bye": {
        "en": "All right. Call me when you want to continue.",
        "zh": "好。需要时再叫我。",
    },
    "chat_who": {
        "en": "I am Luciana. I retrieve the infection forecast, and a solver places exactly six vaccination sites.",
        "zh": "我是 Luciana。我读取感染预报，求解器恰好放置 6 个疫苗接种点。",
    },
    "chat_can": {
        "en": "I can show the forecast, place six vaccination sites after you pick a policy and travel rule, or compare the four policies.",
        "zh": "我可以查看预报、在你选定政策和出行规则后放置 6 个接种点，或比较四种政策。",
    },
    "chat_how": {
        "en": "I am ready whenever you want to continue.",
        "zh": "我这边正常，随时可以继续。",
    },
    "chat_ok": {
        "en": "All right.",
        "zh": "好。",
    },
    "chat_other": {
        "en": "I heard “{snippet}”. I can keep talking, but I stay with the planning task and will not invent sites or years.",
        "zh": "我听到了「{snippet}」。我可以接着聊，但仍围绕规划任务，不会编造点位或年份。",
    },
    "chat_nudge_city": {
        "en": "Please type a city first: **City of Edinburgh** or **Glasgow City**.",
        "zh": "请先告诉我城市：**爱丁堡**或**格拉斯哥**。",
    },
    "chat_nudge_date": {
        "en": "Our latest forecast date is **{latest}**. Tap that option, or type YYYY-MM-DD.",
        "zh": "目前最新预测日是 **{latest}**。可以点下面的按钮，或自己输入 YYYY-MM-DD。",
    },
    "chat_nudge_confirm": {
        "en": "This city still needs a new model. Answer **Yes, build a model** or **No, pick another city**.",
        "zh": "这座城市还需要新建模型。请回答 **是，新建模型** 或 **否，换一座城市**。",
    },
    "chat_nudge_task": {
        "en": "Next I can **plan 6 vaccination sites**, **show the forecast**, or **compare four policies**. Type it, or use the buttons.",
        "zh": "接下来可以 **规划 6 个接种点**、**查看预测**，或 **比较四种政策**。直接打字或点按钮都行。",
    },
    "chat_nudge_zone": {
        "en": "You can also click a neighbourhood on the map, type its name, or pick it from the dropdown.",
        "zh": "也可以在地图上点一个街区、输入名称，或用下拉列表选择。",
    },
    "chat_nudge_scenario": {
        "en": "Please pick a policy first: coverage, equity, preventive, or balanced.",
        "zh": "请先选一种政策：覆盖、公平、预防或均衡。",
    },
    "chat_nudge_travel": {
        "en": "I still need Drive or Walk, and a threshold of 10–30 minutes.",
        "zh": "还需要驾车或步行，以及 10–30 分钟的时限。",
    },
    "chat_nudge_result": {
        "en": "You can also change city, change date, or ask me to explain a neighbourhood.",
        "zh": "你也可以换城市、换日期，或让我解释某个街区。",
    },
    "chat_task_need_city": {
        "en": "I can do that. I still need a city first: **City of Edinburgh** or **Glasgow City**.",
        "zh": "可以。请先告诉我城市：**爱丁堡**或**格拉斯哥**。",
    },
    "chat_date_need_city": {
        "en": "I have **{date}**. Which city should I use — **City of Edinburgh** or **Glasgow City**?",
        "zh": "已记下 **{date}**。请告诉我城市：**爱丁堡**还是**格拉斯哥**？",
    },
    "ask_city": {
        "en": (
            "Which **city** should I use? Type the name in the box. "
            "I currently know **City of Edinburgh** and **Glasgow City**."
        ),
        "zh": "请输入要规划的**城市**。我目前能处理 **City of Edinburgh（爱丁堡）** 和 **Glasgow City（格拉斯哥）**。",
    },
    "ask_date": {
        "en": (
            "Our latest forecast date is **{latest}**. "
            "You can tap that option, type another date (YYYY-MM-DD or 15 June 2022), "
            "or name the city and date together. "
            "I will not run a model until you choose a task."
        ),
        "zh": (
            "我们目前最新的预测日期是 **{latest}**。"
            "可以点下面的「最新预测日」，自己输入日期（如 2022-06-15 或 2022年6月15日），"
            "或一句话说「格拉斯哥 2022年6月15日的预测」。"
            "选定任务之前，我不会运行模型。"
        ),
    },
    "ask_task": {
        "en": (
            "What should I do next?\n\n"
            "The six points are **vaccination sites** — existing GPs, pharmacies, "
            "or mobile stops at car parks. A solver places exactly six of them so "
            "neighbourhoods can reach a site within a travel-time threshold.\n\n"
            "- **Plan 6 vaccination sites** — I place those six vaccination sites. "
            "I first ask which policy (coverage, equity, preventive, or balanced), "
            "then Drive or Walk and the time threshold. Then I show the six points on the map.\n"
            "- **Show the forecast** — I show the predicted infection-rate map, "
            "uncertainty, and how the three graphs mix. I then ask which neighbourhood "
            "to explain with GeoShapley. That explains the forecast, not the site choice.\n"
            "- **Compare four policies** — I keep the same six-site cap and travel rule, "
            "and show how the four policies differ.\n\n"
            "Pick one below, or type it."
        ),
        "zh": (
            "接下来要我做什么？\n\n"
            "这 6 个点是**疫苗接种点**：现有全科诊所、药店，或停车场上的流动接种点。"
            "求解器恰好放 6 个，让街区能在出行时限内到达某个点。\n\n"
            "- **规划 6 个疫苗接种点** — 先问你用哪种政策（覆盖、公平、预防或均衡），"
            "再问驾车/步行和时限，然后在地图上放出这 6 个接种点。\n"
            "- **查看预测** — 展示预测感染率、不确定性和三张图的混合权重，"
            "再问要解释哪个街区。GeoShapley 解释预报，不是选址分数。\n"
            "- **比较四种政策** — 在同样的 6 点上限和出行规则下对比四种政策。\n\n"
            "请点选一项，或直接输入。"
        ),
    },
    "confirm_train": {
        "en": (
            "This city does not have a saved model yet. "
            "Should I build one? Edinburgh's saved model and files will not be overwritten."
        ),
        "zh": "这座城市还没有已保存的模型。是否要新建一个？爱丁堡的模型和文件不会被覆盖。",
    },
    "scope_question": {
        "en": (
            "I have the city-wide forecast. Choose one view first; maps appear only after you pick.\n\n"
            "- **Whole-region view (all Intermediate Zones)** — every neighbourhood on the map, plus the graph mix\n"
            "- **Selected Intermediate Zone view** — pick one Intermediate Zone from the full list, then see that zone\n\n"
            "Type a choice in the box below, or press **Enter** on an empty box to see the two view buttons.\n"
            "GeoShapley explains a zone's forecast; it is not a siting score."
        ),
        "zh": (
            "全市预报已经取回。请先选一种展示方式，选完后才会出现对应结果。\n\n"
            "- **全区域展示（整座城市的全部中间区）** — 所有中间区的地图和图结构混合\n"
            "- **按所选中间区（Intermediate Zone）展示** — 从完整名单里选一个中间区，再看该区结果\n\n"
            "请在下方输入；若想看两种展示按钮，把输入框留空后按 **回车**。\n"
            "GeoShapley 解释该区预报，不是选址分数。"
        ),
    },
    "show_scope_options": {
        "en": "Here are the two views. Click one:",
        "zh": "这是两种展示方式，请点选一项：",
    },
    "show_zone_options": {
        "en": "Here is the full Intermediate Zone list. Pull one neighbourhood:",
        "zh": "这是完整中间区名单，请下拉选择一个街区：",
    },
    "scope_offer": {
        "en": "Choose a view first. The matching maps appear after you select one:",
        "zh": "请先选择展示方式，选完后才会出现对应结果：",
    },
    "scope_need": {
        "en": "Please choose **Whole-region view (all Intermediate Zones)** or **Selected Intermediate Zone view**.",
        "zh": "请选择 **全区域展示（整座城市的全部中间区）** 或 **按所选中间区（Intermediate Zone）展示**。",
    },
    "scope_region_accept": {
        "en": "I will show the whole-region forecast. The maps cover every Intermediate Zone.",
        "zh": "我将展示全区域预报。地图覆盖这座城市的每一个中间区。",
    },
    "btn_scope_region": {
        "en": "Whole-region view (all Intermediate Zones)",
        "zh": "全区域展示（整座城市的全部中间区）",
    },
    "btn_scope_iz": {
        "en": "Selected Intermediate Zone view",
        "zh": "按所选中间区（Intermediate Zone）展示",
    },
    "btn_switch_region": {
        "en": "Back to whole-region view (all Intermediate Zones)",
        "zh": "改回全区域展示（全部中间区）",
    },
    "region_summary": {"en": "Whole-region summary", "zh": "全区域摘要"},
    "region_zones": {"en": "Intermediate Zones", "zh": "中间区数"},
    "region_high_unc": {"en": "High-uncertainty IZs", "zh": "高不确定性中间区"},
    "zone_question": {
        "en": (
            "Type an **Intermediate Zone** name in the box below, or press **Enter** on an empty box "
            "to open the full list.\n\n"
            "Maps and GeoShapley for that zone appear after you choose. "
            "GeoShapley explains the forecast; it is not a siting score."
        ),
        "zh": (
            "请在下方输入一个**中间区（Intermediate Zone）**名称；若想打开完整名单，把输入框留空后按 **回车**。\n\n"
            "选定后才会出现该区地图和 GeoShapley。"
            "GeoShapley 解释该区预报，不是选址分数。"
        ),
    },
    "zone_offer": {
        "en": "Please pull a neighbourhood from the full Intermediate Zone list:",
        "zh": "请从完整中间区名单中下拉选择：",
    },
    "zone_offer_more": {
        "en": "These names matched what you typed. Pick one:",
        "zh": "你输入的名称匹配到多个街区，请选一个：",
    },
    "input_mode_offer": {
        "en": "How do you want to answer?",
        "zh": "请先选一种作答方式：",
    },
    "btn_input_type": {
        "en": "I will type my answer",
        "zh": "自己输入诉求",
    },
    "btn_input_buttons": {
        "en": "Show me Luciana's options",
        "zh": "点选 Luciana 给出的范围",
    },
    "input_mode_type_scenario": {
        "en": "All right. Type a policy: Coverage priority, Equity priority, Preventive priority, or Balanced.",
        "zh": "好。请输入政策：覆盖优先、公平优先、预防优先或均衡。",
    },
    "input_mode_buttons_scenario": {
        "en": "All right. Please click one of the policy buttons I am showing.",
        "zh": "好。请点选我给出的政策按钮。",
    },
    "input_mode_type_travel": {
        "en": "All right. Type a travel mode and a time limit, for example \"walk 20 minutes\" or \"drive 15\".",
        "zh": "好。请输入出行方式和时限，例如「步行 20 分钟」或「驾车 15」。",
    },
    "input_mode_buttons_travel": {
        "en": "All right. Please click Drive or Walk, then one of the minute ranges I am showing (10, 15, 20, 25 or 30).",
        "zh": "好。请点选驾车或步行，再点选我给出的分钟范围（10、15、20、25 或 30）。",
    },
    "input_mode_type_scope": {
        "en": "All right. Type whole-region view, or name one Intermediate Zone.",
        "zh": "好。请输入「全区域展示」，或写出一个中间区名称。",
    },
    "input_mode_buttons_scope": {
        "en": "All right. Please click one of the view buttons I am showing.",
        "zh": "好。请点选我给出的展示方式按钮。",
    },
    "scenario_question": {
        "en": (
            "I can place exactly six vaccination sites. Which policy should I use?\n\n"
            "Type it in the box below, or press **Enter** on an empty box if you want me to show the four policies. "
            "This does not change the forecast."
        ),
        "zh": (
            "我将放置 6 个疫苗接种点。要用哪种政策？\n\n"
            "请在下方输入框填写；若想看我给出的四种政策，把输入框留空后按 **回车**。"
            "这不会改变预报。"
        ),
    },
    "travel_question": {
        "en": (
            "I still need a travel mode and a time threshold before I run the solver.\n\n"
            "Type them below (for example \"walk 20 minutes\"), or press **Enter** on an empty box "
            "to see Drive / Walk and the minute ranges. I will place the six vaccination sites after both are set."
        ),
        "zh": (
            "运行求解器之前，我还需要出行方式和时限。\n\n"
            "请在下方输入（例如「步行 20 分钟」）；若想看我给出的选项，把输入框留空后按 **回车**。"
            "两项都选定后，我再放这 6 个疫苗接种点。"
        ),
    },
    "show_policy_options": {
        "en": "Here are the four policies. Click one:",
        "zh": "这是四种政策，请点选一项：",
    },
    "show_travel_options": {
        "en": "Here are the travel options. Click Drive or Walk, then a time limit:",
        "zh": "这是出行选项。请先点选驾车或步行，再点选时限：",
    },
    "use_city": {"en": "I will use **{city}**. ", "zh": "我将使用 **{city}**。"},
    "use_policy": {"en": "I will use **{policy}**. ", "zh": "我将使用 **{policy}**。"},
    "placing": {
        "en": "I have **{mode}** and **{threshold} min**. I will place six vaccination sites now.",
        "zh": "已记录 **{mode}**、**{threshold} 分钟**。现在放置 6 个疫苗接种点。",
    },
    "city_unknown": {
        "en": "I only know **City of Edinburgh** and **Glasgow City**. Type one of those names.",
        "zh": "我目前只认识 **City of Edinburgh（爱丁堡）** 和 **Glasgow City（格拉斯哥）**。请输入其中一个名称。",
    },
    "city_need": {
        "en": "I still need a city. Type **City of Edinburgh** or **Glasgow City**.",
        "zh": "还需要城市。请输入 **City of Edinburgh** 或 **Glasgow City**（也可写爱丁堡 / 格拉斯哥）。",
    },
    "date_need": {
        "en": "I still need a target date. Type it as YYYY-MM-DD.",
        "zh": "还需要目标日期。请按 YYYY-MM-DD 输入。",
    },
    "date_bad": {
        "en": "Please type a date as YYYY-MM-DD.",
        "zh": "请按 YYYY-MM-DD 输入日期。",
    },
    "confirm_need": {
        "en": "Please answer **Yes, build a model** or **No, pick another city**.",
        "zh": "请回答 **是，新建模型** 或 **否，换一座城市**。",
    },
    "travel_need_mode": {
        "en": "I recorded **{value}**. I still need a time threshold.",
        "zh": "已记录 **{value}**。还需要时间阈值。",
    },
    "travel_need_threshold": {
        "en": "I recorded **{value} min**. I still need Drive or Walk.",
        "zh": "已记录 **{value} 分钟**。还需要选择驾车或步行。",
    },
    "travel_need_both": {
        "en": "I still need {missing}. Pick below, or type them to me.",
        "zh": "还需要{missing}。请点选或直接输入。",
    },
    "policy_need": {
        "en": "Please pick one policy: Coverage-priority, Equity-priority, Preventive-priority, or Balanced. I cannot place sites until you do.",
        "zh": "请选择一种政策：覆盖优先、公平优先、预防优先或均衡。选定之前我不能放置疫苗接种点。",
    },
    "zone_multi": {
        "en": "I found more than one zone: {listed}. Pick one below, or type the full name.",
        "zh": "匹配到多个街区：{listed}。请点选一个，或输入完整名称。",
    },
    "zone_too_many": {
        "en": "That matched {n} zones. Type a longer name, click the map, or pick from the dropdown.",
        "zh": "匹配到 {n} 个街区。请输入更完整的名称、在地图上点选，或用下拉列表选择。",
    },
    "zone_none": {
        "en": "I could not match that Intermediate Zone. Type a name, click it on the map, or pick it from the dropdown.",
        "zh": "无法匹配该中间区。请输入街区名称、在地图上点选，或用下拉列表选择。",
    },
    "zone_accept": {
        "en": "I will explain **{name}**. GeoShapley is for this zone's forecast; it is not a siting score.",
        "zh": "我将解释 **{name}**。GeoShapley 针对该区预报，不是选址分数。",
    },
    "forecast_intro": {
        "en": "Here is the forecast I retrieved.",
        "zh": "这是我取回的预报。",
    },
    "alloc_intro": {
        "en": "Here are the 6 vaccination sites. Click a red point for that site's audit.",
        "zh": "这是 6 个疫苗接种点。点击红点查看该点审计。",
    },
    "compare_intro": {
        "en": "Same 6-site cap and travel constraint. I re-used the recorded scenario runs.",
        "zh": "同样的 6 点上限和出行约束。我复用了已记录的政策结果。",
    },
    "notes_intro": {
        "en": "Graph α and the tool-calling log. α is not a risk share.",
        "zh": "图融合权重 α 与工具调用日志。α 不是风险占比。",
    },
    "no_pipeline": {
        "en": "I have not run a planning task yet. Ask me to plan 6 vaccination sites, or press the button below.",
        "zh": "我还没有做过规划。请让我规划 6 个疫苗接种点，或点下面的按钮。",
    },
    "btn_latest": {"en": "Latest forecast date ({day})", "zh": "最新预测日（{day}）"},
    "btn_plan": {"en": "Plan 6 vaccination sites", "zh": "规划 6 个疫苗接种点"},
    "btn_forecast": {"en": "Show the forecast", "zh": "查看预测"},
    "btn_compare": {"en": "Compare four policies", "zh": "比较四种政策"},
    "btn_train_yes": {"en": "Yes, build a model", "zh": "是，新建模型"},
    "btn_train_no": {"en": "No, pick another city", "zh": "否，换一座城市"},
    "btn_back": {"en": "← Back to Luciana", "zh": "← 返回 Luciana"},
    "btn_home": {"en": "Back to home", "zh": "回到初始主页"},
    "btn_change_policy": {"en": "Change policy or travel", "zh": "更改政策或出行"},
    "btn_another_zone": {
        "en": "Choose another Intermediate Zone",
        "zh": "重新选择中间区（Intermediate Zone）",
    },
    "title_agent": {"en": "Vaccination Site Planning Agent", "zh": "疫苗接种点规划助手"},
    "title_result": {"en": "Vaccination planning result", "zh": "接种点规划结果"},
    "brand_uni": {"en": "University of Glasgow", "zh": "格拉斯哥大学"},
    "brand_sub": {
        "en": "OASIS GeoAI Health Agent",
        "zh": "OASIS GeoAI Health Agent",
    },
    "caption_home_empty": {
        "en": "Type a city and a date in the box. Supported cities and the latest forecast date are listed above.",
        "zh": "请在输入框里填写城市和日期。支持的城市和最新预测日期见上方。",
    },
    "home_coverage": {
        "en": (
            "**Latest forecast date:** {latest}  \n"
            "That day is a 7-day extrapolation from the last observed panel; there is no ground truth on that date.\n\n"
            "**Cities supported now:** City of Edinburgh · Glasgow City"
        ),
        "zh": (
            "**最新预测日期：** {latest}  \n"
            "该日为最后观测日后的 7 日外推，当天没有观测真值。\n\n"
            "**目前支持的城市：** 爱丁堡（City of Edinburgh）· 格拉斯哥（Glasgow City）"
        ),
    },
    "caption_reading": {"en": "Reading: **{city}** · {date}", "zh": "当前：**{city}** · {date}"},
    "continue_agent": {"en": "Continue with Luciana", "zh": "继续与 Luciana 对话"},
    "continue_caption": {
        "en": "Type a follow-up first. The three tasks sit in smaller buttons under the box.",
        "zh": "请先在提问框输入。三个任务按钮在输入框下方，已缩小。",
    },
    "type_box_caption": {
        "en": "Type your answer in Luciana's box:",
        "zh": "请在 Luciana 的提问框里输入：",
    },
    "placeholder_type_scenario": {
        "en": "e.g. Preventive priority…",
        "zh": "例如：预防优先…",
    },
    "placeholder_type_travel": {
        "en": "e.g. walk 20 minutes…",
        "zh": "例如：步行 20 分钟…",
    },
    "placeholder_type_scope": {
        "en": "e.g. whole region, or an Intermediate Zone name…",
        "zh": "例如：全区域，或一个中间区名称…",
    },
    "forecast_maps": {
        "en": "Forecast maps for the view you chose. In Intermediate Zone view you can click another zone on the map, or choose again from the list.",
        "zh": "这是你所选展示方式下的预报图。中间区视图可再点地图换区，或重新从名单选择。",
    },
    "predicted_rate": {"en": "Predicted rate", "zh": "预测感染率"},
    "uncertainty": {"en": "Uncertainty (σ)", "zh": "不确定性（σ）"},
    "uncertainty_flag": {"en": "Uncertainty flag", "zh": "不确定性标记"},
    "coverage": {"en": "Coverage", "zh": "覆盖"},
    "red_points": {
        "en": "Red points are the same 6 vaccination sites as on the uncertainty map.",
        "zh": "红点与不确定性图上是同一批 6 个疫苗接种点。",
    },
    "unverified": {
        "en": "4 March 2023: unverified 7-day extrapolation. No ground truth.",
        "zh": "2023-03-04：未核实的 7 日外推，没有观测真值。",
    },
    "graph_mix": {"en": "Graph mix over time", "zh": "图结构混合随时间变化"},
    "graph_mix_caption": {
        "en": "How much the forecast used the neighbour graph, the road graph, and the commuting graph. These weights are not a risk share and not a siting score.",
        "zh": "预报分别用了多少邻接图、道路图和通勤图。这些权重不是风险占比，也不是选址分数。",
    },
    "graph_mix_saved": {
        "en": "This city has no rolling update trail. The points are the fusion weights stored in the current model, not a made-up time path.",
        "zh": "这座城市没有滚动更新轨迹。图上的点是当前保存模型里的融合权重，不是编造的时间路径。",
    },
    "graph_geo": {"en": "Geographic graph", "zh": "地理邻接图"},
    "graph_transport": {"en": "Transport graph", "zh": "道路交通图"},
    "graph_mobility": {"en": "Mobility graph", "zh": "通勤流动图"},
    "fusion_weight": {"en": "Fusion weight", "zh": "融合权重"},
    "refresh_date": {"en": "Refresh date", "zh": "刷新日期"},
    "geoshapley_title": {"en": "GeoShapley for the selected zone", "zh": "所选街区的 GeoShapley"},
    "geoshapley_caption": {"en": "This explains the forecast, not the site choice.", "zh": "这解释预报，不解释选址。"},
    "zone_label": {"en": "Zone: **{name}**", "zh": "街区：**{name}**"},
    "no_geoshapley": {
        "en": "No GeoShapley table is attached to this city's forecast.",
        "zh": "这座城市的预报没有附带 GeoShapley 表。",
    },
    "no_geoshapley_iz": {"en": "No GeoShapley rows for this IZ.", "zh": "该中间区没有 GeoShapley 记录。"},
    "baseline": {"en": "Baseline", "zh": "基线"},
    "contributions_sum": {"en": "Contributions sum", "zh": "贡献合计"},
    "numbers": {"en": "Numbers and technical breakdown", "zh": "数值与技术分解"},
    "glossary": {"en": "What the labels mean", "zh": "标签含义"},
    "glossary_caption": {
        "en": "Display names only. The solver still uses the recorded field names in the data tables.",
        "zh": "仅用于展示。求解器仍使用数据表中的原字段名。",
    },
    "covered_pop": {"en": "Covered population", "zh": "覆盖人口"},
    "izs_covered": {"en": "IZs covered", "zh": "覆盖中间区"},
    "mean_travel": {"en": "Mean travel time", "zh": "平均出行时间"},
    "max_travel": {"en": "Max travel time", "zh": "最长出行时间"},
    "unserved": {"en": "Unserved zones", "zh": "未覆盖街区"},
    "solver_caption": {
        "en": "Solver output. Population is the last panel zone total. High-uncertainty zones: {n}.",
        "zh": "求解器输出。人口为面板中该区最新总人口。高不确定性街区：{n}。",
    },
    "unc_map_caption": {
        "en": "Forecast uncertainty, not a siting score. Click a red site on either map. High-uncertainty zones: {n}.",
        "zh": "预报不确定性，不是选址分数。可在任一张图上点击红点。高不确定性街区：{n}。",
    },
    "click_site": {
        "en": "Click a vaccination-site button to highlight it and open the audit I retrieved from the solver.",
        "zh": "点击接种点按钮可高亮，并打开求解器给出的审计。",
    },
    "tool_log": {"en": "Tool-calling log", "zh": "工具调用日志"},
    "placeholder_city": {
        "en": "Type a city, or city and date…",
        "zh": "请输入城市，或城市+日期，例如：格拉斯哥 2022-06-15…",
    },
    "placeholder_date": {"en": "Type a date as YYYY-MM-DD…", "zh": "请输入日期 YYYY-MM-DD…"},
    "placeholder_scope": {
        "en": "Type whole region or a neighbourhood name…",
        "zh": "请输入「全区域」或一个街区名称…",
    },
    "placeholder_zone": {
        "en": "Type a neighbourhood name, or use the map / dropdown…",
        "zh": "请输入街区名称，或点地图 / 下拉选择…",
    },
    "zone_dropdown": {"en": "Or pick a neighbourhood from the list", "zh": "或从下拉列表选择街区"},
    "zone_dropdown_placeholder": {"en": "Select a neighbourhood…", "zh": "请选择街区…"},
    "neighbourhood": {"en": "Neighbourhood", "zh": "街区"},
    "iz_code": {"en": "Zone code", "zh": "街区代码"},
    "placeholder_ask": {
        "en": "e.g. I want the Glasgow forecast for 2022-06-15…",
        "zh": "例如：我想看格拉斯哥 2022-06-15 的预测…",
    },
    "spinner": {"en": "Loading the forecast and placing 6 vaccination sites...", "zh": "正在读取预报并放置 6 个疫苗接种点…"},
    "policy_coverage": {"en": "Coverage priority", "zh": "覆盖优先"},
    "policy_equity": {"en": "Equity priority", "zh": "公平优先"},
    "policy_preventive": {"en": "Preventive priority", "zh": "预防优先"},
    "policy_balanced": {"en": "Balanced", "zh": "均衡"},
    "travel_drive": {"en": "Drive", "zh": "驾车"},
    "travel_walk": {"en": "Walk", "zh": "步行"},
    "covered": {"en": "Covered", "zh": "已覆盖"},
    "unserved_label": {"en": "Unserved", "zh": "未覆盖"},
    "assigned_site": {"en": "Assigned site", "zh": "分配接种点"},
    "travel_time": {"en": "Travel time (min)", "zh": "出行时间（分钟）"},
    "allocated_sites": {"en": "Vaccination sites", "zh": "疫苗接种点"},
    "flag_high": {"en": "high", "zh": "高"},
    "flag_normal": {"en": "Normal", "zh": "正常"},
    "btn_send": {"en": "Send", "zh": "发送"},
    "label_agent": {"en": "Luciana", "zh": "Luciana"},
    "label_min": {"en": "min", "zh": "分钟"},
    "policy_line": {
        "en": "Policy **{policy}** · {threshold:.0f} min {travel}.",
        "zh": "政策 **{policy}** · {threshold:.0f} 分钟 {travel}。",
    },
    "raises": {"en": "Raises forecast", "zh": "推高预报"},
    "lowers": {"en": "Lowers forecast", "zh": "拉低预报"},
    "contribution": {"en": "Contribution to predicted rate", "zh": "对预测感染率的贡献"},
    "indicator": {"en": "Indicator", "zh": "指标"},
    "direction": {"en": "Direction", "zh": "方向"},
    "glossary_name": {"en": "Name", "zh": "名称"},
    "glossary_meaning": {"en": "Meaning", "zh": "含义"},
    "col_scenario": {"en": "Scenario", "zh": "政策"},
    "col_covered_iz": {"en": "Covered IZs", "zh": "覆盖中间区"},
    "col_coverage_pct": {"en": "Coverage %", "zh": "覆盖率"},
    "col_covered_pop": {"en": "Covered Pop", "zh": "覆盖人口"},
    "col_mean_tt": {"en": "Mean Travel Time", "zh": "平均出行时间"},
    "col_max_tt": {"en": "Max Travel Time", "zh": "最长出行时间"},
    "col_unserved_iz": {"en": "Unserved IZs", "zh": "未覆盖中间区"},
    "geo_baseline_help": {
        "en": "Predicted rate if this zone's SIMD indicators and location were replaced by this city's reference medians. Neighbours stay as observed.",
        "zh": "若将该区的 SIMD 指标和区位换成该市参考中位数后的预测感染率。邻区保持观测值。",
    },
    "geo_rate_help": {
        "en": "Model forecast for this zone (rolling 7-day rate per 100,000).",
        "zh": "该区模型预报（每 10 万人滚动 7 日感染率）。",
    },
    "geo_sum_help": {
        "en": "Sum of the indicator bars. Baseline + this sum equals the predicted rate.",
        "zh": "各指标条之和。基线加上该合计等于预测感染率。",
    },
    "geo_baseline_info": {
        "en": "**Baseline** is not a city-average infection rate. It is the model's prediction for **this zone** after swapping its SIMD values and its map location for this city's reference medians. The bars below then show how this zone's real values push the forecast from Baseline up or down to the Predicted rate.",
        "zh": "**基线**不是全市平均感染率。它是把该区 SIMD 和区位换成该市参考中位数后，模型对该区的预测。下方柱状图显示真实值如何把预报从基线推到预测感染率。",
    },
    "geo_scope": {"en": "Local explanation for the selected zone", "zh": "针对所选街区的局部解释"},
    "site_gp": {"en": "GP", "zh": "全科诊所"},
    "site_pharmacy": {"en": "Pharmacy", "zh": "药店"},
    "site_mobile": {"en": "Mobile stop (car park)", "zh": "流动点（停车场）"},
}

FEATURE_ZH = {
    "baseline": "基线",
    "location": "区位",
    "income_deprivation": "收入剥夺",
    "employment_deprivation": "就业剥夺",
    "higher_education": "高等教育",
    "overcrowding": "过度拥挤",
    "crime": "犯罪",
    "public_transport_time_to_gp": "公交到全科诊所时间",
    "location_x_income_deprivation": "区位 × 收入剥夺",
    "location_x_employment_deprivation": "区位 × 就业剥夺",
    "location_x_higher_education": "区位 × 高等教育",
    "location_x_overcrowding": "区位 × 过度拥挤",
    "location_x_crime": "区位 × 犯罪",
    "location_x_public_transport_time_to_gp": "区位 × 公交到全科诊所时间",
}

GLOSSARY_ZH = [
    ("预测感染率", "滚动 7 日每 10 万人 COVID-19 预测感染率。不是每日新发病例。"),
    ("不确定性（σ）", "该感染率的预测标准差。σ 越大，预报越不确定。"),
    ("不确定性标记", "若 σ 高于校准的第 90 百分位则为高，否则为正常。这是地图叠加，不是选址分数。"),
    ("街区 / 中间区", "2011 Intermediate Zone。爱丁堡有 111 个街区。地图按区代码连接。"),
    ("全科诊所", "作为候选疫苗接种点的 NHS 全科诊所。"),
    ("药店", "作为候选疫苗接种点的社区药店。"),
    ("流动点（停车场）", "OSM 停车场，用作可能的流动/临时接种点。"),
    ("驾车 / 步行", "OSM 路网上的出行方式。驾车按 30 km/h，步行按 4.5 km/h。"),
    ("出行时限", "若最近已选点在该分钟数内，则该中间区视为已覆盖。"),
    ("覆盖优先", "在时限内覆盖尽可能多的人口，选出 6 个疫苗接种点。"),
    ("公平优先", "优先收入剥夺高、当前服务不足的街区（SIMD 收入剥夺率与公交到全科诊所时间）。"),
    ("预防优先", "优先预测感染率高和/或不确定性高的街区。"),
    ("均衡", "综合覆盖、公平与预防热点权重。"),
    ("基线", "将该区六项 SIMD 指标和区位设为全市中位数后的预测感染率。邻区保持观测值。不是全市平均感染率。"),
    ("收入剥夺", "SIMD 收入剥夺人口占比。用于解释预报，不用作选址分数。"),
    ("就业剥夺", "该区 SIMD 就业剥夺指标。"),
    ("高等教育", "SIMD 高等教育/升学指标。"),
    ("过度拥挤", "SIMD 过度拥挤指标。"),
    ("犯罪", "SIMD 街区犯罪指标。"),
    ("公交到全科诊所时间", "SIMD 公交到全科诊所的分钟数。这是现状可达性，不是我们的 OSM 出行矩阵。"),
    ("区位", "命名特征之后的残差空间贡献（东/北坐标）。"),
    ("区位 × …", "区位与该特征的交互。以固定图结构为条件。"),
    ("地理 / 交通 / 通勤图", "三张图的学习融合权重（α）。是图源相对重要性，不是 COVID 风险占比。"),
    ("所选街区的局部解释", "此处 GeoShapley 针对目标街区局部解释，不是因果推断。"),
]

SCENARIO_I18N = {
    "coverage priority": "policy_coverage",
    "coverage-priority": "policy_coverage",
    "coverage": "policy_coverage",
    "equity priority": "policy_equity",
    "equity-priority": "policy_equity",
    "equity": "policy_equity",
    "preventive priority": "policy_preventive",
    "preventive-priority": "policy_preventive",
    "preventive": "policy_preventive",
    "balanced": "policy_balanced",
}


def has_chinese(text: str) -> bool:
    return bool(_CJK.search(str(text or "")))


_INTERNAL_ENGLISH = {
    "plan 6 sites",
    "show the forecast",
    "compare the four allocation policies.",
    "show the allocation",
}
_CITY_ONLY = re.compile(
    r"^(city of edinburgh|edinburgh|glasgow city|glasgow|s12000036|s12000049)$",
    re.I,
)
_DATE_ONLY = re.compile(r"^20\d{2}-\d{2}-\d{2}(?:\s+\(.*\))?$")


def detect_ui_lang(text: str) -> str | None:
    """Return zh/en when the typed line shows a language, else None (keep current)."""
    raw = str(text or "").strip()
    if not raw:
        return None
    if has_chinese(raw):
        return "zh"
    if raw.lower() in _INTERNAL_ENGLISH:
        return None
    if _DATE_ONLY.match(raw) or _CITY_ONLY.match(raw):
        return None
    if not re.search(r"[A-Za-z]{2,}", raw):
        return None
    return "en"


def t(lang: str, key: str, **kwargs) -> str:
    row = STRINGS.get(key) or {}
    out = row.get(lang) or row.get("en") or key
    if kwargs:
        return out.format(**kwargs)
    return out


def scenario_display(lang: str, english_label: str) -> str:
    key = SCENARIO_I18N.get(str(english_label or "").strip().lower().replace("-", " "))
    if not key:
        key = SCENARIO_I18N.get(str(english_label or "").strip().lower())
    if key:
        return t(lang, key)
    return english_label


def travel_display(lang: str, english_label: str) -> str:
    raw = str(english_label or "").strip().lower()
    if raw in {"walk", "步行"}:
        return t(lang, "travel_walk")
    return t(lang, "travel_drive")


def city_display(lang: str, name: str) -> str:
    raw = str(name or "")
    lowered = raw.lower()
    if lang != "zh":
        return raw
    if "glasgow" in lowered or "格拉斯哥" in raw:
        return "格拉斯哥（Glasgow City）"
    if "edinburgh" in lowered or "爱丁堡" in raw:
        return "爱丁堡（City of Edinburgh）"
    return raw


def feature_display(lang: str, key: str, fallback: str | None = None) -> str:
    if lang == "zh" and key in FEATURE_ZH:
        return FEATURE_ZH[key]
    return fallback or key


def site_type_display(lang: str, english_label: str) -> str:
    mapping = {
        "GP": "site_gp",
        "Pharmacy": "site_pharmacy",
        "Mobile stop (car park)": "site_mobile",
    }
    key = mapping.get(english_label)
    if key:
        return t(lang, key)
    return english_label


def glossary_rows(lang: str, english_rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
    if lang == "zh":
        return list(GLOSSARY_ZH)
    return list(english_rows)
