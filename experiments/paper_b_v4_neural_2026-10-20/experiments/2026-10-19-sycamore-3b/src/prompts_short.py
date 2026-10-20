"""Templated open-ended short-answer prompts (several plausible answers, real quality differences)."""
import itertools
T = {
 'Suggest a name for {}.': ['a new bakery in a small town', 'a pet golden retriever', 'a coffee shop near a university', 'a startup that sells reusable water bottles',
                            'a fantasy kingdom', 'a robot vacuum cleaner', 'a jazz band', 'a hiking club', 'a children\'s bookstore', 'a vegan restaurant',
                            'a sailboat', 'a podcast about history', 'a mobile app for budgeting', 'a black cat', 'a family farm'],
 'Write a slogan for {}.': ['a gym', 'a recycling campaign', 'an electric car brand', 'a public library', 'a pizza delivery service', 'a sunscreen brand',
                            'a math tutoring service', 'a city marathon', 'a bicycle shop', 'an online language school'],
 'Give a title for {}.': ['a novel about a lighthouse keeper', 'a song about leaving home', 'a documentary about bees', 'a poem about autumn rain',
                          'a mystery set on a train', 'a memoir of a chef', 'a science fiction film about Mars', 'a painting of a stormy sea',
                          'a blog post about saving money', 'a short story about a lost key'],
 'Recommend one {}.': ['book for someone who loves adventure', 'movie for a rainy evening', 'hobby for a retired teacher', 'board game for a family night',
                       'city to visit in Europe', 'fruit to eat after exercise', 'houseplant for beginners', 'instrument for a child to learn',
                       'dish to cook for a first date', 'sport for someone who dislikes running'],
 'Name {}.': ['a famous scientist', 'a color that feels calm', 'an animal that symbolizes courage', 'a word that sounds happy', 'a classic novel',
              'a flower to give a friend', 'a planet you would like to visit', 'a historical figure you admire', 'a dessert for a birthday party',
              'a superpower you would like to have'],
 'Describe {} in a few words.': ['the ocean at night', 'a perfect weekend', 'the smell of rain', 'your ideal job', 'a busy train station',
                                 'the first day of spring', 'a good friend', 'a mountain village', 'the taste of coffee', 'a quiet library'],
}
def short_prompts():
    out = []
    for tpl, fills in T.items():
        for f in fills:
            out.append(tpl.format(f))
    return out
if __name__ == '__main__':
    print(len(short_prompts()))
