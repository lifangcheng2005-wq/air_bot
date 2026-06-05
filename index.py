import os
import requests
from flask import Flask, request, jsonify
from google import genai  # 最新版 Google GenAI SDK
from google.genai import types

app = Flask(__name__)

# ==================== ⚙️ 全域初始化與環境變數撈取 ====================
# 💡 安全讀取 Vercel 後台的 NEW_GEMINI_KEY
NEW_KEY = os.environ.get("NEW_GEMINI_KEY", "")
client = genai.Client(api_key=NEW_KEY)

CWA_API_KEY = os.environ.get("CWA_API_KEY", "")
MOENV_API_KEY = os.environ.get("MOENV_API_KEY", "")

CITIES = ['臺北', '台北', '新北', '桃園', '臺中', '台中', '臺南', '台南', '高雄', '基隆', 
          '新竹', '苗栗', '彰化', '南投', '雲林', '嘉義', '屏東', '宜蘭', '花蓮', '台東', '臺東', '澎湖', '金門', '連江']

def parse_target_city(user_text: str) -> str:
    """自動從使用者字串中過濾並對齊政府 Open Data 的標準縣市名稱"""
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


def fetch_government_data(city_name: str) -> str:
    """即時向中央氣象署與環境部爬取最新的觀測與預報資料（防禦安全版）"""
    pure_city = city_name.replace("市", "").replace("縣", "")
    data_context = f"【查詢目標縣市】: {city_name}\n"
    
    # 1. 氣象預報 (F-C0032-001)
    try:
        cwa_url = f"https://opendata.cwa.gov.tw/api/v1/rest/datastore/F-C0032-001?Authorization={CWA_API_KEY}&locationName={city_name}"
        cwa_res = requests.get(cwa_url, timeout=2.5).json()
        if 'records' in cwa_res and cwa_res['records']['location']:
            elements = cwa_res['records']['location'][0]['weatherElement']
            wx = elements[0]['time'][0]['parameter']['parameterName']
            pop = elements[1]['time'][0]['parameter']['parameterName']
            min_t = elements[2]['time'][0]['parameter']['parameterName']
            max_t = elements[4]['time'][0]['parameter']['parameterName']
            data_context += f"[中央氣象署天氣預報]：天氣狀態為 {wx}，降雨機率 {pop}%，氣溫介於 {min_t}°C 至 {max_t}°C。\n"
        else:
            data_context += "[中央氣象署天氣預報]：暫時找不到該縣市的天氣預報數據。\n"
    except Exception:
        data_context += "[中央氣象署天氣預報]：氣象署連線逾時，改由大方向預報支援。\n"

    # 2. 空氣品質(AQI) 與 紫外線(UVI) -> 使用 .get() 語法，防止 KeyError 暴斃
    try:
        data_context += "[環境部環境觀測]：\n"
        
        # 空氣品質 AQI (限縮 limit 減少傳輸時間防止超時)
        aqi_url = f"https://data.moenv.gov.tw/api/v2/aqx_p_43?api_key={MOENV_API_KEY}&limit=50&format=JSON"
        aqi_res = requests.get(aqi_url, timeout=2.8).json()
        aqi_records = aqi_res.get('records', [])
        
        city_aqi = next((item for item in aqi_records if pure_city in item.get('county', '') or city_name in item.get('county', '')), None)
        
        if city_aqi:
            data_context += f"- AQI 空氣品質指標：{city_aqi.get('aqi', '無資料')} ({city_aqi.get('status', '無資料')})，PM2.5 濃度為 {city_aqi.get('pm2.5', '無資料')} μg/m³。\n"
        else:
            data_context += "- 空氣品質：該地區測站目前連線維護中。\n"
            
        # 紫外線 UVI
        uv_url = f"https://data.moenv.gov.tw/api/v2/uv_p_01?api_key={MOENV_API_KEY}&limit=40&format=JSON"
        uv_res = requests.get(uv_url, timeout=2.5).json()
        uv_records = uv_res.get('records', [])
        
        city_uv = next((item for item in uv_records if pure_city in item.get('county', '') or city_name in item.get('county', '')), None)
        
        if city_uv:
            data_context += f"- 紫外線指數 (UVI)：{city_uv.get('uvi', '無資料')}，分級狀態為 ({city_uv.get('status', '無資料')})。\n"
        else:
            data_context += "- 紫外線指數：目前無即時觀測數據。\n"
            
    except Exception:
        data_context += "- 環境部觀測站數據目前連線過載，暫時跳過。\n"

    return data_context


@app.route('/webhook', methods=['POST'])
def dialogflow_webhook():
    """同步接收並解析 Dialogflow Webhook JSON 格式"""
    req_data = request.get_json(force=True, silent=True)
    if not req_data:
        return jsonify({"fulfillmentText": "🤖 Webhook 沒有收到有效的 JSON 請求。"})

    query_result = req_data.get('queryResult', {})
    action = query_result.get('action', '')
    user_message = query_result.get('queryText', '')
    parameters = query_result.get('parameters', {})

    # 雙重管道撈取城市，防止 Dialogflow 帶入雜質參數
    target_city_param = parameters.get('geo-city', '')
    if target_city_param:
        target_city = parse_target_city(target_city_param)
    else:
        target_city = parse_target_city(user_message)

    # 執行爬蟲撈取即時政府開放資料
    live_government_info = fetch_government_data(target_city)
    
    instruction_text = f"""
    你是一個親切貼心的「生活環境氣象智慧助理」。
    請根據下方由系統提供、當前最新從「交通部中央氣象署」與「環境部」Open Data 爬取而來的資料，回答使用者的提問。
    
    【當前政府 Open Data 即時數據庫】:
    {live_government_info}
    
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
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"使用者目前的提問是：{user_message if