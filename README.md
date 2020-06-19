[![LinkedIn][linkedin-shield]](https://www.linkedin.com/in/advaithasabnis/)
[![Website][website-shield]](https://advait.herokuapp.com/)

<br />
<p align="center">
  <a href="https://github.com/advaithasabnis/insight">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>
  <h3 align="center">Target and Market</h3></p>

  <p align="center">
    A Propensity Modelling Project
    <br />
    by Advait Hasabnis
    <br />
    <a href="https://advait.herokuapp.com/" target="_blank"><strong>Explore the dashboard »</strong></a>
</p>



<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Data](#data)
* [Approach](#approach)
* [Built With](#built-with)
* [Contact](#contact)

<!-- CONTENTS -->
## About The Project

This project is in collaboration with a company that develops a popular cryptocurrency portfolio tracking application for Android and iOS. As of May 2020, it has hundreds of thousands of monthly active users. The app has a freemium model and users have the option to purchase its paid premium service. The app does not store any personal information about its users such as their email addresses or phone numbers. As a result, targeted marketing efforts are impeded by the lack of demographic information.

<b>Target and Market</b> uses anonymized user behaviour and in-app data to identify the best candidates (that are most likely to convert) for a marketing campaign. The tool further segments the selected targets to enable the company to maximize their return on marketing investment by tailoring their campaign to each segment.

<!-- DATA -->
## Data
Data for this project is private. Every user's events in the app are logged via Google Analytics for Firebase and stored on Google BigQuery. This project uses events from May 2020 with over 150 million logged events. For every logged event, a timestamp, event_type, session id and engagement time are available. Additionally, data such as the number of transactions and value of investment portfolio are available via a flat file.

At the request of the company, numbers have been scaled using a secret multiplier to obfuscate real values.

<!-- APPROACH -->
## Approach
<ul>
<li>Data: Query and clean required data from BigQuery. Merge with data from .csv files.</li>
<li>Engineer features: average session time, recency (last session), days of activity, value of investments, number of transactions, geolocation.</li>
<li>Use XGBoost to predict premium users. Assign a probability to each user.</li>
<li>Build a dashboard that allows the company to select top N users (with highest probability of being premium)</li>
<li>Use a clustering algorithm to segment selected users</li>
</ul>

<!-- TOOLS AND FRAMEWORKS -->
## Built With
* [BigQuery](https://cloud.google.com/bigquery/)
* [Pandas](https://pandas.pydata.org/)
* [Scikit-Learn](https://scikit-learn.org/stable/index.html)
* [Plotly Dash](https://plotly.com/dash/)
* [Heroku](https://www.heroku.com/)

<!-- CONTACT -->
## Author
<p><b>Advait Hasabnis</b></p>

Project Link: [https://github.com/advaithasabnis/target-and-market](https://github.com/advaithasabnis/target-and-market)
<br>
Dashboard Link: [https://advait.herokuapp.com/](https://advait.herokuapp.com/)


<!-- MARKDOWN LINKS & IMAGES -->
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat&logo=linkedin&colorB=2867B2
[website-shield]: https://img.shields.io/badge/-Website-blueviolet?style=flat
