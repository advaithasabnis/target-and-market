<p align="center">
  <h3 align="center">Target and Market</h3>

  <p align="center">
    A Customer Segmentation Project
    <br />
    by Advait Hasabnis
    <br />
    <a href="https://github.com/advaithasabnis/"><strong>Explore the docs »</strong></a>
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

[Delta](https://delta.app) is a popular cryptocurrency portfolio tracking application for Android and iOS. As of May 2020, it has hundreds of thousands of monthly active users. Delta has a freemium model and users have the option to purchase its paid service, Delta Pro.

Target and Market uses anonymized user behaviour data for the app to help Delta better understand their users and tailor their marketing efforts to monetize more users.

<!-- DATA -->
## Data

Data for this project is private. Every user's events in the app are logged via Google Analytics for Firebase and stored on Google BigQuery. This project uses events from May 2020 with over 150 million logged events. 

## Approach

[![Customer Segmentation][clustering-scheme]](https://github.com/advaithasabnis/insight)

* Query required data from BigQuery
* Engineer features: total engagement time, frequency of interaction, recency
* Cluster users
* Validate clusters by visualization and lift in terms of ratio of paid users

## Built With
* [BigQuery](https://cloud.google.com/bigquery/)
* [Pandas](https://pandas.pydata.org/)
* [Scikit-Learn](https://scikit-learn.org/stable/index.html)

<!-- CONTACT -->
## Contact

Advait Hasabnis - advait.iitb@gmail.com

Project Link: [https://github.com/advaithasabnis/insight](https://github.com/advaithasabnis/insight)

<!-- MARKDOWN LINKS & IMAGES -->
[clustering-scheme]: images/clustering_scheme.png