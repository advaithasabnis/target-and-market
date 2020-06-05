[![LinkedIn][linkedin-shield]](https://www.linkedin.com/in/advaithasabnis/)

<br />
<p align="center">
  <h3 align="center">Target and Market</h3>

  <p align="center">
    A Propensity Modelling Project
    <br />
    by Advait Hasabnis
    <br />
    <a href="https://github.com/advaithasabnis/insight"><strong>Explore the docs »</strong></a>
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

<b>Target and Market</b> uses anonymized user behaviour and in-app data to help the company better understand their users and tailor their marketing efforts to monetize more users.

<!-- DATA -->
## Data
Data for this project is private. Every user's events in the app are logged via Google Analytics for Firebase and stored on Google BigQuery. This project uses events from May 2020 with over 150 million logged events. For every logged event, a timestamp, event_type, session id and engagement time are available.

At the request of the company, numbers have been scaled using a multiplier to obfuscate real numbers. 

<!-- APPROACH -->
## Approach
<ul>
<li>Data: Query required data from BigQuery. Merge with data from .csv file.</li>
<li>Engineer features: average session time, date of first install, recency (last session), value of investments, geolocation</li>
<li>Use XGBoost to predict premium users</li>
<li>Identify false positives as potential customers to target convert to paid users</li>
<li>Build a dashboard using Dash to visualize the results</li>
</ul>

<!-- TOOLS AND FRAMEWORKS -->
## Built With
* [BigQuery](https://cloud.google.com/bigquery/)
* [Pandas](https://pandas.pydata.org/)
* [Scikit-Learn](https://scikit-learn.org/stable/index.html)
* [Plotly Dash](https://plotly.com/dash/)

<!-- CONTACT -->
## Author
Advait Hasabnis
Project Link: [https://github.com/advaithasabnis/insight](https://github.com/advaithasabnis/insight)

<!-- MARKDOWN LINKS & IMAGES -->
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat&logo=linkedin&colorB=2867B2