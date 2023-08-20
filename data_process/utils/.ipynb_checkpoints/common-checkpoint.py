import json
import requests


def httpPost_response(url, dict_data, headers={'Content-type': 'application/json'}):
    '''
    params:
        url - string, hattp address to the website
        dict_data - dictionary, contain the parameters needed by the query
        
    return:
        reponse - result get from the query
    '''
    json_data = json.dumps(dict_data)
    # Send the POST request with JSON data
    headers = headers
    response = requests.post(url, data=json_data, headers=headers)
    return response

def get_pdbSeq_GraphQL(list_entryID, url_graphql='https://data.rcsb.org/graphql',
                      query='''
                              query($id: [String!]!){
                              entries(entry_ids: $id)
                              {
                                rcsb_id
                                polymer_entities {
                                  rcsb_polymer_entity_container_identifiers {
                                    entity_id
                                    asym_ids
                                    auth_asym_ids
                                    reference_sequence_identifiers {
                                      database_name
                                      database_accession
                                    }
                                  }
                                  entity_poly {
                                    pdbx_seq_one_letter_code_can
                                    rcsb_sample_sequence_length
                                    type
                                  }
                                  rcsb_polymer_entity {
                                    formula_weight
                                  }
                                }
                              }
                            }
                            '''):
    '''
    params:
        list_entryID - list, e.g. ["101M","102L","102M","103L","103M","104L","104M"]
        url_graphql - check https://data.rcsb.org/index.html#rest-api for more infomation
        query - $id indicates the entry_IDs wish to download. includes all features as well. check https://data.rcsb.org/index.html#rest-api
        
    return:
        string, request results. Json result but in String.
    '''
    # Prepare the JSON data
    data = {
      #  'query': '{entry(entry_id:"4HHB"){exptl{method}}}'
         "query": query,
        # "query": "query($id: String!){entry(entry_id:$id){exptl{method}}}",
        "variables": {"id": list_entryID}
    }
    # Send the POST request with JSON data
    headers = {'Content-type': 'application/json'}
    
    response = httpPost_response(url=url_graphql, dict_data=data, headers=headers)
    
    return response.text