from neo4j import GraphDatabase
import os
import logging


def import_got_data(uri, username, password):
    driver = GraphDatabase.driver(uri, auth=(username, password))

    with driver.session() as session:
        # Create constraints separately
        session.run("""
        CALL apoc.schema.assert(
        {Location:['name'],Region:['name']},
        {Battle:['name'],Person:['name'],House:['name']});        
        """)
        
        session.run("""
        LOAD CSV WITH HEADERS FROM 
        "https://raw.githubusercontent.com/tomasonjo/neo4j-game-of-thrones/master/data/battles.csv" as row
        //merge node labeled Battle 
        MERGE (b:Battle{name:row.name})
        ON CREATE SET b.year = toInteger(row.year),
                    b.summer = row.summer,
                    b.major_death = row.major_death,
                    b.major_capture = row.major_capture,
                    b.note = row.note,
                    b.battle_type = row.battle_type,
                    b.attacker_size = toInteger(row.attacker_size),
                    b.defender_size = toInteger(row.defender_size)
        """)

        session.run("""
        LOAD CSV WITH HEADERS FROM 
        "https://raw.githubusercontent.com/tomasonjo/neo4j-game-of-thrones/master/data/battles.csv" as row
        // there is only attacker_outcome in the data, 
        // so we do a CASE statement for defender_outcome
        WITH row,
        case when row.attacker_outcome = "win" THEN "loss" ELSE "win" END as defender_outcome
        // match the battle
        MATCH (b:Battle{name:row.name})
        // all battles have atleast one attacker so we don't have to use foreach trick
        MERGE (attacker1:House{name:row.attacker_1}) 
        MERGE (attacker1)-[a1:ATTACKER]->(b) 
        ON CREATE SET a1.outcome = row.attacker_outcome

        // When we want to skip null values we can use foreach trick
        FOREACH
        (ignoreMe IN CASE WHEN row.defender_1 is not null THEN [1] ELSE [] END | 
            MERGE (defender1:House{name:row.defender_1})
            MERGE (defender1)-[d1:DEFENDER]->(b)
            ON CREATE SET d1.outcome = defender_outcome)
        FOREACH
        (ignoreMe IN CASE WHEN row.defender_2 is not null THEN [1] ELSE [] END | 
            MERGE (defender2:House{name:row.defender_2})
            MERGE (defender2)-[d2:DEFENDER]->(b)
            ON CREATE SET d2.outcome = defender_outcome)
        FOREACH
        (ignoreMe IN CASE WHEN row.attacker_2 is not null THEN [1] ELSE [] END | 
            MERGE (attacker2:House{name:row.attacker_2})
            MERGE (attacker2)-[a2:ATTACKER]->(b)
            ON CREATE SET a2.outcome = row.attacker_outcome)
        FOREACH
        (ignoreMe IN CASE WHEN row.attacker_3 is not null THEN [1] ELSE [] END | 
            MERGE (attacker2:House{name:row.attacker_3})
            MERGE (attacker3)-[a3:ATTACKER]->(b)
            ON CREATE SET a3.outcome = row.attacker_outcome)
        FOREACH
        (ignoreMe IN CASE WHEN row.attacker_4 is not null THEN [1] ELSE [] END | 
            MERGE (attacker4:House{name:row.attacker_4})
            MERGE (attacker4)-[a4:ATTACKER]->(b)
            ON CREATE SET a4.outcome = row.attacker_outcome)
        """)

        session.run("""
        LOAD CSV WITH HEADERS FROM 
        "https://raw.githubusercontent.com/tomasonjo/neo4j-game-of-thrones/master/data/battles.csv" as row
        MATCH (b:Battle{name:row.name})
        // We use coalesce, so that null values are replaced with "Unknown" 
        MERGE (location:Location{name:coalesce(row.location,"Unknown")})
        MERGE (b)-[:IS_IN]->(location)
        MERGE (region:Region{name:row.region})
        MERGE (location)-[:IS_IN]->(region)
        """)

        session.run("""
        LOAD CSV WITH HEADERS FROM 
        "https://raw.githubusercontent.com/tomasonjo/neo4j-game-of-thrones/master/data/battles.csv" as row
        // We split the columns that may contain more than one person
        WITH row,
            split(row.attacker_commander,",") as att_commanders,
            split(row.defender_commander,",") as def_commanders,
            split(row.attacker_king,"/") as att_kings,
            split(row.defender_king,"/") as def_kings,
            row.attacker_outcome as att_outcome,
            CASE when row.attacker_outcome = "win" THEN "loss" 
            ELSE "win" END as def_outcome
        MATCH (b:Battle{name:row.name})
        // we unwind a list
        UNWIND att_commanders as att_commander
        MERGE (p:Person{name:trim(att_commander)})
        MERGE (p)-[ac:ATTACKER_COMMANDER]->(b)
        ON CREATE SET ac.outcome=att_outcome
        // to end the unwind and correct cardinality(number of rows)
        // we use any aggregation function ( e.g. count(*))
        WITH b,def_commanders,def_kings,att_kings,att_outcome,def_outcome,count(*) as c1
        UNWIND def_commanders as def_commander
        MERGE (p:Person{name:trim(def_commander)})
        MERGE (p)-[dc:DEFENDER_COMMANDER]->(b)
        ON CREATE SET dc.outcome = def_outcome
        // reset cardinality with an aggregation function (end the unwind)
        WITH b,def_kings,att_kings,att_outcome,def_outcome,count(*) as c2
        UNWIND def_kings as def_king
        MERGE (p:Person{name:trim(def_king)})
        MERGE (p)-[dk:DEFENDER_KING]->(b)
        ON CREATE SET dk.outcome = def_outcome
        // reset cardinality with an aggregation function (end the unwind)
        WITH b,att_kings,att_outcome,count(*) as c3
        UNWIND att_kings as att_king
        MERGE (p:Person{name:trim(att_king)})
        MERGE (p)-[ak:ATTACKER_KING]->(b)
        ON CREATE SET ak.outcome = att_outcome
        """)

        session.run("""
        LOAD CSV WITH HEADERS FROM 
        "https://raw.githubusercontent.com/tomasonjo/neo4j-game-of-thrones/master/data/character-deaths.csv" as row
        return count(row.Allegiances)
        """)

        session.run("""
        LOAD CSV WITH HEADERS FROM 
        "https://raw.githubusercontent.com/tomasonjo/neo4j-game-of-thrones/master/data/character-deaths.csv" as row
        MATCH (h:House{name:row.Allegiances})
        return count(row.Allegiances)
        """)

        session.run("""
        LOAD CSV WITH HEADERS FROM 
        "https://raw.githubusercontent.com/tomasonjo/neo4j-game-of-thrones/master/data/character-deaths.csv" as row
        // use a replace function to remove "House "
        MATCH (h:House{name:replace(row.Allegiances,"House ","")})
        return count(row.Allegiances)
        """)

        session.run("""
        LOAD CSV WITH HEADERS FROM 
        "https://raw.githubusercontent.com/tomasonjo/neo4j-game-of-thrones/master/data/character-deaths.csv" as row
        // we can use CASE in a WITH statement
        with row,
            case when row.Nobility = "1" then "Noble" else "Commoner" end as status_value
        // as seen above we remove "House " for better linking
        MERGE (house:House{name:replace(row.Allegiances,"House ","")})
        MERGE (person:Person{name:row.Name})
        // we can also use CASE statement inline
        SET person.gender = case when row.Gender = "1" then "male" else "female" end,
            person.book_intro_chapter = row.`Book Intro Chapter`, 
            person.book_death_chapter = row.`Death Chapter`,
            person.death_year = toInteger(row.`Death Year`)
        MERGE (person)-[:BELONGS_TO]->(house)
        MERGE (status:Status{name:status_value})
        MERGE (person)-[:HAS_STATUS]->(status)
        // doing the foreach trick to skip null values
        FOREACH
        (ignoreMe IN CASE WHEN row.GoT = "1" THEN [1] ELSE [] END | 
            MERGE (book1:Book{sequence:1}) 
            ON CREATE SET book1.name = "Game of thrones" 
            MERGE (person)-[:APPEARED_IN]->(book1))
        FOREACH
        (ignoreMe IN CASE WHEN row.CoK = "1" THEN [1] ELSE [] END | 
            MERGE (book2:Book{sequence:2}) 
            ON CREATE SET book2.name = "Clash of kings" 
            MERGE (person)-[:APPEARED_IN]->(book2))
        FOREACH
        (ignoreMe IN CASE WHEN row.SoS = "1" THEN [1] ELSE [] END | 
            MERGE (book3:Book{sequence:3}) 
            ON CREATE SET book3.name = "Storm of swords" 
            MERGE (person)-[:APPEARED_IN]->(book3))
        FOREACH
        (ignoreMe IN CASE WHEN row.FfC = "1" THEN [1] ELSE [] END | 
            MERGE (book4:Book{sequence:4}) 
            ON CREATE SET book4.name = "Feast for crows" 
            MERGE (person)-[:APPEARED_IN]->(book4))
        FOREACH
        (ignoreMe IN CASE WHEN row.DwD = "1" THEN [1] ELSE [] END | 
            MERGE (book5:Book{sequence:5}) 
            ON CREATE SET book5.name = "Dance with dragons" 
            MERGE (person)-[:APPEARED_IN]->(book5))
        FOREACH
        (ignoreMe IN CASE WHEN row.`Book of Death` is not null THEN [1] ELSE [] END | 
            MERGE (book:Book{sequence:toInteger(row.`Book of Death`)}) 
            MERGE (person)-[:DIED_IN]->(book))
        """)

        session.run("""
        LOAD CSV WITH HEADERS FROM 
        "https://raw.githubusercontent.com/tomasonjo/neo4j-game-of-thrones/master/data/character-predictions.csv" as row
        MERGE (p:Person{name:row.name})
        // set properties on the person node
        SET p.title = row.title,
            p.death_year = toInteger(row.DateoFdeath),
            p.birth_year = toInteger(row.dateOfBirth),
            p.age = toInteger(row.age),
            p.gender = case when row.male = "1" then "male" else "female" end
        // doing the foreach trick to skip null values
        FOREACH
        (ignoreMe IN CASE WHEN row.mother is not null THEN [1] ELSE [] END |
            MERGE (mother:Person{name:row.mother})
            MERGE (p)-[:RELATED_TO{name:"mother"}]->(mother)
        )
        FOREACH
        (ignoreMe IN CASE WHEN row.spouse is not null THEN [1] ELSE [] END |
            MERGE (spouse:Person{name:row.spouse})
            MERGE (p)-[:RELATED_TO{name:"spouse"}]->(spouse)
        )
        FOREACH
        (ignoreMe IN CASE WHEN row.father is not null THEN [1] ELSE [] END |
            MERGE (father:Person{name:row.father})
            MERGE (p)-[:RELATED_TO{name:"father"}]->(father)
        )
        FOREACH
        (ignoreMe IN CASE WHEN row.heir is not null THEN [1] ELSE [] END |
            MERGE (heir:Person{name:row.heir})
            MERGE (p)-[:RELATED_TO{name:"heir"}]->(heir)
        )
        // we remove "House " from the value for better linking of data
        FOREACH 
        (ignoreMe IN CASE WHEN row.house is not null THEN [1] ELSE [] END | 
            MERGE (house:House{name:replace(row.house,"House ","")}) 
            MERGE (p)-[:BELONGS_TO]->(house) 
        )
        """)

        session.run("""
        LOAD CSV WITH HEADERS FROM 
        "https://raw.githubusercontent.com/tomasonjo/neo4j-game-of-thrones/master/data/character-predictions.csv" as row
        // match person
        MERGE (p:Person{name:row.name})
        // doing the foreach trick... we lower row.culture for better linking
        FOREACH
        (ignoreMe IN CASE WHEN row.culture is not null THEN [1] ELSE [] END |
            MERGE (culture:Culture{name:lower(row.culture)})
            MERGE (p)-[:MEMBER_OF_CULTURE]->(culture)
        )
        FOREACH
        (ignoreMe IN CASE WHEN row.book1 = "1" THEN [1] ELSE [] END |
            MERGE (book:Book{sequence:1})
            MERGE (p)-[:APPEARED_IN]->(book)
        )
        FOREACH
        (ignoreMe IN CASE WHEN row.book2 = "1" THEN [1] ELSE [] END |
            MERGE (book:Book{sequence:2})
            MERGE (p)-[:APPEARED_IN]->(book)
        )
        FOREACH
        (ignoreMe IN CASE WHEN row.book3 = "1" THEN [1] ELSE [] END |
            MERGE (book:Book{sequence:3})
            MERGE (p)-[:APPEARED_IN]->(book)
        )
        FOREACH
        (ignoreMe IN CASE WHEN row.book4 = "1" THEN [1] ELSE [] END |
            MERGE (book:Book{sequence:4})
            MERGE (p)-[:APPEARED_IN]->(book)
        )
        FOREACH
        (ignoreMe IN CASE WHEN row.book5 = "1" THEN [1] ELSE [] END |
            MERGE (book:Book{sequence:5})
            MERGE (p)-[:APPEARED_IN]->(book)
        )
        """)

        session.run("""
        LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/tomasonjo/neo4j-game-of-thrones/master/data/character-predictions.csv" as row
        // do CASE statements
        with row,
        case when row.isAlive = "0" THEN [1] ELSE [] END as dead_person,
        case when row.isAliveMother = "0" THEN [1] ELSE [] END as dead_mother,
        case when row.isAliveFather = "0" THEN [1] ELSE [] END as dead_father,
        case when row.isAliveHeir = "0" THEN [1] ELSE [] END as dead_heir,
        case when row.isAliveSpouse = "0" THEN [1] ELSE [] END as dead_spouse
        // MATCH all the persons
        MATCH (p:Person{name:row.name})
        // We use optional match so that it doesnt stop the query if not found
        OPTIONAL MATCH (mother:Person{name:row.mother})
        OPTIONAL MATCH (father:Person{name:row.father})
        OPTIONAL MATCH (heir:Person{name:row.heir})
        OPTIONAL MATCH (spouse:Spouse{name:row.spouse})
        // Set the label of the dead persons
        FOREACH (d in dead_person | set p:Dead)
        FOREACH (d in dead_mother | set mother:Dead)
        FOREACH (d in dead_father | set father:Dead)
        FOREACH (d in dead_heir | set heir:Dead)
        FOREACH (d in dead_spouse | set spouse:Dead)
        """)

        session.run("""
        MATCH (p:Person) where p.death_year IS NOT NULL
        SET p:Dead
        """)

        session.run("""
        MATCH (p:Person)-[:DEFENDER_KING|ATTACKER_KING]-()
        SET p:King
        """)

        session.run("""
        MATCH (p:Person) where p.title = "Ser"
        SET p:Knight
        """)


    driver.close()



def main():    
    load_dotenv('.env')
    
    uri = os.environ["NEO4J_URI"]
    username = os.environ["NEO4J_USERNAME"]
    password = os.environ["NEO4J_PASSWORD"]
    
    import_got_data(uri, username, password)
    print("Import completed successfully!")

if __name__ == "__main__":
    main()