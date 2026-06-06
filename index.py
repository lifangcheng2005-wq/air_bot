import os
import requests
from flask import Flask, request, jsonify
from google import genai  # 最新版 Google GenAI SDK
from google.genai import types
from concurrent.futures import ThreadPoolExecutor  # 💡 導入平行多執行緒工具

app = Flask(__name__)

# ==================== ⚙️ 全域初始化與環境變數撈取 ====================
NEW_KEY = os.environ.get("NEW_GEMINI_KEY", "")
client = genai.Client(api_key=NEW_KEY)

CWA_API_KEY = os.environ.get("CWA_API_KEY", "")
MOENV_API_KEY = os.environ.get("MOENV_API_KEY", "")

CITIES = ['臺北', '台北', '新北', '桃園', '臺中', '台中', '臺南', '台南', '高雄', '基隆', 
          '新竹', '苗栗', '彰化', '南投', '雲林', '嘉義', '屏東', '宜蘭', '花蓮', '台東', '臺東', '澎湖', '金門', '連江']

def parse_target_city(user_text: str) -> str:
    if not user_text:
        return "臺北市"
    matched_city = "臺北市"  
    for city in CITIES:
        if city in user_text:
            matched_city = city
            break
    if matched_city.startswith("台"):
        matched_city = "臺" + matched_city[1:]
    if not matched_city.endswith(("市", "縣")):
        if matched_city in ['臺北', '新北', '桃園', '臺中', '臺南', '高雄', '基隆', '新竹', '嘉義']:
            matched_city += "市"
        else:
            matched_city += "縣"
    return matched_city

# 💡 拆分為獨立的爬蟲任務，準備進行平行發送
def get_cwa_data(city_name):
    try:
        cwa_url = f"https://opendata.cwa.gov.tw/api/v1/rest/datastore/F-C0032-001?Authorization={CWA_API_KEY}&locationName={city_name}"
        res = requests.get(cwa_url, timeout=2.0).json()
        if 'records' in res and res['records']['location']:
            elements = res['records']['location'][0]['weatherElement']
            return {
                "wx": elements[0]['time'][0]['parameter']['parameterName'],
                "pop": elements[1]['time'][0]['parameter']['parameterName'],
                "min_t": elements[2]['time'][0]['parameter']['parameterName'],
                "max_t": elements[4]['time'][0]['parameter']['parameterName']
            }
    except Exception:
        pass
    return None

def get_moenv_data(pure_city, city_name):
    try:
        # 空氣品質 AQI
        alt_city = pure_city.replace("臺", "台") if "臺" in pure_city else pure_city.replace("台", "臺")
        aqi_url = f"https://data.moenv.gov.tw/api/v2/aqx_p_43?api_key={MOENV_API_KEY}&limit=60&format=JSON"
        aqi_res = requests.get(aqi_url, timeout=2.0).json()
        aqi_records = aqi_res.get('records', [])
        city_aqi = next((item for item in aqi_records if item.get('county') in [pure_city, alt_city, city_name]), None)
        
        # 紫外線 UVI
        uv_url = f"https://data.moenv.gov.tw/api/v2/uv_p_01?api_key={MOENV_API_KEY}&limit=40&format=JSON"
        uv_res = requests.get(uv_url, timeout=2.0).json()
        uv_records = uv_res.get('records', [])
        city_uv = next((item for item in uv_records if item.get('county') in [pure_city, alt_city, city_name]), None)
        
        return {"aqi": city_aqi, "uv": city_uv}
    except Exception:
        pass
    return None

def fetch_government_data(city_name: str) -> str:
    """【終極平行版】同時派兩組人馬去抓資料，時間省一半！"""
    pure_city = city_name.replace("市", "").replace("縣", "")
    data_context = f"【查詢目標縣市】: {city_name}\n"
    
    # 💡 使用 ThreadPoolExecutor 讓氣象署與環境部 API 同步進行並行發送！
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_cwa = executor.submit(get_cwa_data, city_name)
        future_moenv = executor.submit(get_moenv_data, pure_city, city_name)
        
        cwa_result = future_cwa.result()
        moenv_result = future_moenv.result()

    # 組合氣象數據
    if cwa_result:
        data_context += f"[中央氣象署天氣預報]：天氣狀態為 {cwa_result['wx']}，降雨機率 {cwa_result['pop']}%，氣溫介於 {cwa_result['min_t']}°C 至 {cwa_result['max_t']}°C。\n"
    else:
        data_context += "[中央氣象署天氣預報]：氣象署線路忙碌中。\n"

    # 組合環境部數據
    data_context += "[環境部環境觀測]：\n"
    if moenv_result:
        aqi = moenv_result.get("aqi")
        uv = moenv_result.get("uv")
        if aqi:
            data_context += f"- AQI 空氣品質指標：{aqi.get('aqi', '無資料')} ({aqi.get('status', '無資料')})，PM2.5 濃度為 {aqi.get('pm2.5', '無資料')} μg/m³。\n"
        else:
            data_context += "- 空氣品質：該地區目前無即時監測指標。\n"
        if uv:
            data_context += f"- 紫外線指數 (UVI)：{uv.get('uvi', '無資料')}，分級狀態為 ({uv.get('status', '無資料')})。\n"
        else:
            data_context += "- 紫外線指數：目前無即時觀測數據。\n"
    else:
        data_context += "- 環境部觀測站數據目前連線過載，暫時跳過。\n"

    return data_context


@app.route('/webhook', methods=['POST'])
def dialogflow_webhook():
    req_data = request.get_json(force=True, silent=True)
    if not req_data:
        return jsonify({"fulfillmentText": "🤖 Webhook 沒有收到有效的 JSON 請求。"})

    query_result = req_data.get('queryResult', {})
    user_message = query_result.get('queryText', '')
    parameters = query_result.get('parameters', {})

    target_city = parse_target_city(parameters.get('geo-city', '') if parameters.get('geo-city', '') else user_message)

    # 執行超高速平行爬蟲
    live_government_info = fetch_government_data(target_city)
    
    instruction_text = """
    你是一個親切貼心的「生活環境氣象智慧助理」。
    請根據使用者提供給妳的最新政府 Open Data 即時數據，用有條理、親切口語化的方式回答提問。
    
    【回答規則】:
    1. 必須一律使用「繁體中文 (zh-TW)」回答。
    2. 說話風格要親切、口語化、有條理。
    3. 依據天氣或環境狀況，適度加入適當的 emoji (例：🌤, 🌧, 😷)，讓訊息易於閱讀。
    4. 嚴格遵守數據庫內容，不要胡亂捏造數值。
    """
    
    try:
        ai_config = types.GenerateContentConfig(
            max_output_tokens=500,
            system_instruction=instruction_text
        )
        
        ai_prompt = f"""
        【當前政府 Open Data 即時數據庫】:
        {live_government_info}
        
        【使用者目前的提問】: {user_message if user_message else target_city}
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=ai_prompt,
            config=ai_config,
        )
        ai_reply_text = response.text if response.text else "抱歉，我現在無法生成回應，請稍後再試。"
        
    except Exception as e:
        # 如果白天 Google 免費版真的排隊太久，依然交由備用純後端真數據直接播報
        ai_reply_text = f"🤖 報時氣象台（AI 臨時塞車，改由後端直接播報即時數據）：\n\n{live_government_info}"

    return jsonify({"fulfillmentText": ai_reply_text})
