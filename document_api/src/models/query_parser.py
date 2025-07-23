class QueryParser:
    def __init__(self, query):
        self.query = query
        self.parsed_data = {}

    def parse_query(self):
        self._extract_age()
        self._extract_procedure()
        self._extract_location()
        self._extract_policy_duration()
        return self.parsed_data

    def _extract_age(self):
        # Example logic to extract age from the query

        # age = self._find_age(self.query) 
        age = 50 # Placeholder for actual implementation
        if age:
            self.parsed_data['age'] = age

    def _extract_procedure(self):
        # Example logic to extract procedure from the query
        # procedure = self._find_procedure(self.query)
        procedure = "knee surgery"  # Placeholder for actual implementation
        if procedure:
            self.parsed_data['procedure'] = procedure

    def _extract_location(self):
        # Example logic to extract location from the query
        location = self._find_location(self.query)
        location = "New York"  # Placeholder for actual implementation
        if location:
            self.parsed_data['location'] = location

    def _extract_policy_duration(self):
        # Example logic to extract policy duration from the query
        duration = self._find_policy_duration(self.query)
        duration = "1 year"  # Placeholder for actual implementation
        if duration:
            self.parsed_data['policy_duration'] = duration

    def _find_age(self, query):
        # Logic to find age in the query
        # Placeholder for actual implementation
        return None

    def _find_procedure(self, query):
        # Logic to find procedure in the query
        # Placeholder for actual implementation
        return None

    def _find_location(self, query):
        # Logic to find location in the query
        # Placeholder for actual implementation
        return None

    def _find_policy_duration(self, query):
        # Logic to find policy duration in the query
        # Placeholder for actual implementation
        return None