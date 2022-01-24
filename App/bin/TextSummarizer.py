from textblob import TextBlob

sentiment = TextBlob("Therefore, leakage of the fluid from the connection part of the outer cylinders (or connection parts of the inner cylinders) is unlikely to occur in this connector, and especially in the case in which a cryogenic fluid such as liquefied hydrogen or the like is being handled, the reliability of such a connector is high in the case of use in applications in which heat shrinkage may occur in the outer cylinders or inner cylinders").sentiment

print(sentiment)