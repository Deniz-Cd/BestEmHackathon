from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
import time
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor

def get_ua(filename):
	with open(filename, "r", encoding="utf-8") as f:
		user_agents = [line.strip() for line in f if line.strip()]
	random_ua = random.choice(user_agents)
	print("Random User-Agent:", random_ua)
	return random_ua


def search_city(city, country):	
	options = Options()

	options.add_argument("--start-maximized")
	options.add_argument("--headless=new")
	options.add_argument("--disable-gpu")
	options.add_argument("--disable-infobars")
	options.add_argument("--disable-images")
	options.add_argument("--mute-audio")
	options.add_argument("--ignore-certificate-errors")
	options.add_argument("--disable-blink-features=AutomationControlled")
	options.add_experimental_option("excludeSwitches", ["enable-automation"])
	options.add_experimental_option("useAutomationExtension", False)
	options.add_argument("--lang=en-US")
	options.page_load_strategy = 'eager'

	
	random_ua = get_ua("user_agents.txt")
	options.add_argument(f'user-agent={random_ua}')

	city = city.replace(" ", "-")
	country = country.replace(" ", "-")
	
	driver = webdriver.Chrome(options=options)
	try:
		# Prima încercare: doar oraș
		driver.get(f"https://www.numbeo.com/crime/in/{city}")
		time.sleep(0.2)
		city_exists = False
		try:
			element = driver.find_element(By.CSS_SELECTOR, "div.innerWidth > h1")
			if "Cannot find city" in element.text:
				print(f"{city} not found, trying with country...")
				# A doua încercare: oraș + țară
				driver.get(f"https://www.numbeo.com/crime/in/{city}-{country}")
				time.sleep(0.2)
				try:
					element = driver.find_element(By.CSS_SELECTOR, "div.innerWidth > h1")
					if "Cannot find city" in element.text:
						print(f"{city}, {country} does not have any available data.")
					else:
						print(f"{city}, {country} found!")
						city_exists = True
				except NoSuchElementException:
					print(f"{city}, {country} found!")
					city_exists = True

			else:
				print(f"{city} found!")
				city_exists = True

		except NoSuchElementException:
			print(f"{city} found!")
			city_exists = True

	finally:
		if city_exists == False:
			driver.quit()

	data = []

	rows = driver.find_elements(By.CSS_SELECTOR, "table.table_builder_with_value_explanation.data_wide_table:nth-of-type(1) tbody tr")
	for row in rows:
		try:
			category = row.find_element(By.CSS_SELECTOR, "td.columnWithName").text.strip()
			value = row.find_element(By.CSS_SELECTOR, "td.indexValueTd").text.strip()
			data.append({"category": category, "value": value})
		except:
			continue

	rows = driver.find_elements(By.CSS_SELECTOR, "table.table_builder_with_value_explanation.data_wide_table:nth-of-type(2) tbody tr")
	for row in rows:
		try:
			category = row.find_element(By.CSS_SELECTOR, "td.columnWithName").text.strip()
			value = row.find_element(By.CSS_SELECTOR, "td.indexValueTd").text.strip()
			data.append({"category": category, "value": value})
		except:
			continue

	try:
		contributors_elem = driver.find_element(By.CSS_SELECTOR, "span.reportees")
		contributors_text = contributors_elem.text.strip()
		data.append({"category": "Contributors", "value": contributors_text})
	except NoSuchElementException:
		contributors_text = None

	driver.quit() 

	with open(f"{city}_{country}.json", "w", encoding="utf-8") as file:
		json.dump(data, file, ensure_ascii=False, indent=4)


cities = [
    ("Slatina", "Romania"),
    ("Paris", "France"),
    ("London", "UK"),
    ("New York", "USA"),
]

def worker(city_country):
    city, country = city_country
    try:
        search_city(city, country)
    except Exception as e:
        print(f"Error scraping {city}, {country}: {e}")

with ThreadPoolExecutor(max_workers=10) as executor:
    executor.map(worker, cities)