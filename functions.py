from datetime import datetime
import webbrowser as wb


def check_time():
    current_time = datetime.now().time()
    return f"Сейчас {current_time.hour} {current_time.minute}"


def web_browser(input):
    wb.open('https://www.google.com/search?q=' + input)
