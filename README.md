# Overall task

The task is to develop an algorithm to disambiguate people contributing to various scientific events (e.g. talks, presentations, sessions).

For this, you have to create one profile per person. This profile will have all the contribution from that person (100% recall) and no other contributions from anyone else (100% precision).

Please avoid unnecessary duplicates as well as mixing contributions from different scientists despite similar names/focus-areas.

For the purposes, you have to use following data:

1. data.json: List of 5086 various contributions, described by several attributes (features), e.g. names, information about the workplace of the author, its geolocation, and focus areas (key topics covered in contribution)

2. ground_truth.json: "Ground truth" - actual group s of contributions from the data file (each contribution is assigned to a person)

3. persons.json: The list of unique people.

The solution should be an algorithm or model which allows us to disambiguate people based on the data.json. With that in mind, please complete the following points:

• Analyze the data

• Build a model and provide a (minimal) solution

• Measure the performance of the model (F1, precision, recall, etc)