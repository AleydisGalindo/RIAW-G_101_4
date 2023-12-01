import json
import random
import pickle


class AnalyticsData:
    """
    An in memory persistence object.
    Declare more variables to hold analytics tables.
    """

    def save_query_terms(self, terms: str) -> int:
        print(self)
        return random.randint(0, 100000)

    ## STATS
    fact_clicks_stats = dict([])        # fact_clicks is a dictionary with the click counters: key = doc id | value = click counter
    fact_algorithms_stats = dict([])    # fact_algorithms is a dictionary with the selected algorithm counters: key = selected_algorithm | value = algorithm counter
    fact_terms_stats = dict([])         # fact_terms is a dictionary with the query terms counters: key = query term | value = term counter
    fact_agents_stats = dict([])        # fact_agents is a dictionary with the user agent info: key = agent | value = {'platform': {...}, 'os': {...}, 'bot': False, 'browser': {...}}
    fact_query_sizes_stats = dict([])   # fact_query_sizes is a dictionary with the query size counters: key = terms | value = queries
    fact_city_stats = dict([])          # fact_city is a dictionary with the ip city counters: key = city | value = counter of the number of ips from a city
    fact_dwell_time_stats = dict([])    # fact_dwell_time is a dictionary with the dwell time counters: key = dwell time | value = counter of how many times a dwell time happens
    fact_week_days_stats = dict([])     # fact_week_days is a dictionary with the day of the week counters: key = day of the week | value = counter of the number of users using the web in that day of the week
    fact_ip_stats = dict([])            # fact_ip is a dictionary with the ip counters: key = ip | value = counter of the number of queries this ip does

    ## DASHBOARD
    try:
        with open('dashboard_data.pkl', 'rb') as file:
            data = pickle.load(file)

        fact_clicks = data['fact_clicks']
        fact_algorithms = data['fact_algorithms']
        fact_terms = data['fact_terms']
        fact_agents = data['fact_agents']
        fact_query_sizes = data['fact_query_sizes']
        fact_city = data['fact_city']
        fact_dwell_time = data['fact_dwell_time']
        fact_week_days = data['fact_week_days']
        fact_ip = data['fact_ip']

    except:
        fact_clicks = dict([])
        fact_algorithms = dict([])
        fact_terms = dict([])
        fact_agents = dict([])
        fact_ip = dict([])
        fact_query_sizes = dict([])
        fact_city = dict([])
        fact_dwell_time = dict([])
        fact_week_days = dict([])

        data = {"fact_clicks": fact_clicks, "fact_algorithms": fact_algorithms, "fact_terms": fact_terms, "fact_agents": fact_agents, "fact_ip": fact_ip, "fact_query_sizes": fact_query_sizes, "fact_city": fact_city, "fact_dwell_time": fact_dwell_time, "fact_week_days": fact_week_days}
        with open('dashboard_data.pkl', 'wb') as file:
            pickle.dump(data, file) 



class ClickedDoc:
    def __init__(self, doc_id, description, counter):
        self.doc_id = doc_id
        self.description = description
        self.counter = counter

    def to_json(self):
        return {
            'doc_id': self.doc_id,
            'description': self.description,
            'counter': self.counter
        } # self.__dict__

    def __str__(self):
        """
        Print the object content as a JSON string
        """
        return json.dumps(self)
