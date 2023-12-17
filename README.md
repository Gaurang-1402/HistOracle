
> [!NOTE]  
> Don't forget to add your own credials to the `.env` file. Setup the SSO config in IAM while configuring the GCP datastore.


![image.png](https://user-images.githubusercontent.com/48355572/214252830-b5c764db-25c2-451a-b74c-876423f81917.png)

## Inspiration 💡
_**History**_, a crucial element in shaping one's identity, often comes off as dull and uninteresting. Understanding the importance of communicating history correctly, especially in the context of identity, motivated us to explore innovative ways to make history engaging.

Thus we came up with **HistOracle**! ✨

![image.png](https://i.postimg.cc/xd8nQ7LW/image.png)


## What it does 🤔
HistOracle transforms the exploration of historical figures and monuments into an interactive and delightful experience. The app caters not only to students but also to museums seeking to attract tourists worldwide. During our recent trip to Puerto Rico, we encountered the common challenge of grasping the historical significance of statues and monuments. HistOracle emerged as a solution to this problem, allowing users to interact directly with historical figures and learn about their stories.


## How we built it ⚙️
We built the interface using React.js and MaterialUI. After a seamless Single Sign-On (SSO) experience, HistOracle's home dashboard appears, featuring nearby statues and historical monuments. For privacy, we derive your approximate location from your public IP, displaying relevant monuments on a Leaflet.js-powered map. 


![image.png](https://i.postimg.cc/9QyMDskK/image.png)

OpenAI's ChatGPT API scrapes metadata, including images, for historical sites. We've created a pipeline connecting these with preprocessed audio from Elevenlabs. Using the Audio-Visual Correlation Transformer (AVCT), we generate one-shot talking face animations, inferring motions from visual images and sample audio. Our architecture uses gRPC for high-performance APIs in microservices, with a Dockerized backend. The ChatGPT-powered chatbot works seamlessly! 


#### **Submission Category Age group** → **18+**
#### **Submission Track** → **Communication** 💬

## Usage

1. Start the Docker containers:
  ```
    docker-compose up
  ```
2. The application will be accessible at:
  ```
    localhost:3333
  ```



## Challenges we ran into 😤
With last-minute brainstorming just two hours before the hackathon and exams looming, creativity was a challenge. We opted for a minimalistic approach, emphasizing that less is more. Time constraints and the need to keep the interface simple presented additional challenges.



## Accomplishments that we're proud of ✨
Despite the challenges, we successfully developed HistOracle, providing an innovative solution to the problem of making history engaging. The app's unique features, such as one-shot talking face animations and interactive conversations with historical figures, showcase our team's creativity and determination.


## What we learned 🙌
Through the development of HistOracle, we learned the importance of simplicity in design, especially when faced with time constraints. The integration of various technologies, from React.js to AVCT, expanded our understanding of creating engaging and interactive applications.


## What's next for HistOracle? 🚀
Beyond the hackathon, our plans include incorporating new features to cater to a wider audience. We aim to gather feedback to enhance the user interface and expand the app's reach globally.

**Note ⚠️ — API credentials have been revoked. If you want to run the same on your local, use your own credentials.**
