NBA All-Star Predictor: Can data spot elite talent? 🏀

I built this project to see if a Machine Learning model could actually distinguish an "All-Star" performance from a standard one using over 30 years of NBA historical data.

Most fans look at points, but I wanted to see how much weight the "hidden" stats like blocks, turnovers, and physical attributes (height/weight) actually carry when the AI makes a decision.
🚀 Try it yourself

You can test the model here: [https://nba-allstar-predictor-fkjtypgiy4cqmrv6cayqsy.streamlit.app/#nba-all-star-performance-predictor]
🧠 How it works (The Techy Stuff)

Instead of just a simple "Yes/No", the model gives a probability score

    Model: XGBoost Classifier (chosen for its speed and accuracy with tabular data).

    Dataset: 330,000+ individual game records.

    The "Threshold" Logic: After some testing, I set the decision threshold at 25%. In the NBA, being even 25% more likely to be an outlier than a regular player is often what separates the elite from the rest.

📊 What does the AI care about?

I generated a feature importance plot to see what's happening under the hood. It turns out, the model is quite realistic—it doesn't just fall for "high points" but looks at the overall efficiency
