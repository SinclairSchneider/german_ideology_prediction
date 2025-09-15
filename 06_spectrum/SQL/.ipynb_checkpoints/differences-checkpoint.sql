SELECT avg(difference) FROM 
(
	SELECT id_statement, abs(arrayDifference(groupArray(toInt32(translate(substring(id_answer, 1, 1),'012', '201'))))[2])/2 as difference FROM bundestagdb.wahlomat
	WHERE party_name in ['DIE LINKE', 'PDS', 'DIE LINKE.', 'DIE LINKE.PDS', 'Die Linke'] OR
	party_name in ['GRÜNE', 'BÜNDNIS 90/DIE GRÜNEN', 'Bündnis 90/Die Grünen', 'GRÜNE/B 90', 'GRÜNE/GAL', 'Die Grünen', 'Bündnis 90/ Die Grünen']
	GROUP BY id_statement
	HAVING length(groupArray(id_answer)) = 2
)

SELECT avg(difference) FROM 
(
	SELECT id_statement, abs(arrayDifference(groupArray(toInt32(translate(substring(id_answer, 1, 1),'012', '201'))))[2])/2 as difference FROM bundestagdb.wahlomat
	WHERE party_name in ['GRÜNE', 'BÜNDNIS 90/DIE GRÜNEN', 'Bündnis 90/Die Grünen', 'GRÜNE/B 90', 'GRÜNE/GAL', 'Die Grünen', 'Bündnis 90/ Die Grünen'] OR
	party_name in ['SPD']
	GROUP BY id_statement
	HAVING length(groupArray(id_answer)) = 2
)

SELECT avg(difference) FROM 
(
	SELECT id_statement, abs(arrayDifference(groupArray(toInt32(translate(substring(id_answer, 1, 1),'012', '201'))))[2])/2 as difference FROM bundestagdb.wahlomat
	WHERE party_name in ['SPD'] OR
	party_name in ['FDP']
	GROUP BY id_statement
	HAVING length(groupArray(id_answer)) = 2
)

SELECT avg(difference) FROM 
(
	SELECT id_statement, abs(arrayDifference(groupArray(toInt32(translate(substring(id_answer, 1, 1),'012', '201'))))[2])/2 as difference FROM bundestagdb.wahlomat
	WHERE party_name in ['CDU', 'CSU', 'CDU/CSU', 'CDU / CSU', 'Freie Union'] OR
	party_name in ['FDP']
	GROUP BY id_statement
	HAVING length(groupArray(id_answer)) = 2
)

SELECT avg(difference) FROM 
(
	SELECT id_statement, abs(arrayDifference(groupArray(toInt32(translate(substring(id_answer, 1, 1),'012', '201'))))[2])/2 as difference FROM bundestagdb.wahlomat
	WHERE party_name in ['CDU', 'CSU', 'CDU/CSU', 'CDU / CSU', 'Freie Union'] OR
	party_name in ['AfD', 'NPD']
	GROUP BY id_statement
	HAVING length(groupArray(id_answer)) = 2
)

SELECT avg(difference) FROM 
(
	SELECT id_statement, abs(arrayDifference(groupArray(toInt32(translate(substring(id_answer, 1, 1),'012', '201'))))[2])/2 as difference FROM bundestagdb.wahlomat
	WHERE party_name in ['FDP'] OR
	party_name in ['GRÜNE', 'BÜNDNIS 90/DIE GRÜNEN', 'Bündnis 90/Die Grünen', 'GRÜNE/B 90', 'GRÜNE/GAL', 'Die Grünen', 'Bündnis 90/ Die Grünen']
	GROUP BY id_statement
	HAVING length(groupArray(id_answer)) = 2
)

SELECT avg(difference) FROM 
(
	SELECT id_statement, abs(arrayDifference(groupArray(toInt32(translate(substring(id_answer, 1, 1),'012', '201'))))[2])/2 as difference FROM bundestagdb.wahlomat
	WHERE party_name in ['SPD'] OR
	party_name in ['DIE LINKE', 'PDS', 'DIE LINKE.', 'DIE LINKE.PDS', 'Die Linke']
	GROUP BY id_statement
	HAVING length(groupArray(id_answer)) = 2
)