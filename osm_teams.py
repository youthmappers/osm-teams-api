import pandas as pd
import requests, sys, json


class OSMTeams():
    
    def __init__(self, token_or_session, organization_id, per_page=50, debug=False):
        self.API_URL = 'https://mapping.team/api/'
        self.ORG = organization_id
        self.PER_PAGE = per_page

        self.debug = debug
        
        if type(token_or_session) is not str:
            self.session = token_or_session
        else:
            self.session = requests.Session()
            self.session.headers = {'Content-Type':'application/json','Authorization': 'Bearer {}'.format(token_or_session)}
            
    
    def _handle_pages(self, pagination, url, per_page=None, max_pages=None):
        if self.debug:
            max_pages = 2
        last_page = max_pages or pagination.get('lastPage')
        data = []

        for page in range(2, last_page+1):
            res = self.session.get(url + f'?page='+str(page) + (f'&perPage={per_page}' if per_page else ''))
            data.append(res.json())
        return data
            
    def get_team(self, team):
        res = self.session.get(self.API_URL + f"teams/{team}")
        return res.json()
    
    
    def get_team_members(self, team):
        _page1 = self.session.get(self.API_URL + f"teams/{team}/members").json()
        
        if 'members' in _page1:
            data = _page1.get('members').get('data')
            
            if 'pagination' in _page1.get('members'):
                pagination = _page1.get('members').get('pagination')
                
                remaining_pages = self._handle_pages( pagination, self.API_URL + f"teams/{team}/members" )
                
                for page in remaining_pages:
                    data += page.get('members').get('data')
                
                if not self.debug:
                    assert len(data) == pagination.get('total'), "Pagination returned incorrect number of team members"
        
        return data

    #Not paginated
    def get_team_moderators(self, team):
        res = self.session.get(self.API_URL + f"teams/{team}/moderators")
        return res.json()

    
    def get_team_attributes(self, team):
        res = self.session.get(self.API_URL + f'profiles/teams/{team}')
        if res.text == 'Not Found' or res.status_code == 404:
            return {}
        result = res.json()
        try:
            if len(result) > 0:
                return dict([(x.get('name'), x.get('value')) for x in result])
            else:
                return {}
        except:
            print(self.API_URL + f'profiles/teams/{team}')
            print(result)
            raise
            
    
    def get_all_organization_teams(
        self, 
        attributes=False, 
        members=False,
        join_link=False, 
        max_count=False,
        per_page=10
    ):
        """
        1. Gets the list of teams from the organizations/:id/teams endpoint.
        2. Iterates over the teams, collecting the user IDs, attributes, and join link if requested
        
        Returns: Dataframe of all teams with attributes
        """
        
        print("\nFetching all organization teams", end="")


        url = self.API_URL + 'organizations/' + str(self.ORG) + '/teams'
        
        _page1 = self.session.get(url + f"?page=1&perPage={per_page}").json()
        
        if 'data' in _page1:
            data = _page1.get('data')
            
            if 'pagination' in _page1:
                pagination = _page1.get('pagination')

                print(f" - paginating {pagination.get('lastPage'):,} pages for {pagination.get('total'):,} teams")
                
                remaining_pages = self._handle_pages(pagination, url, per_page=per_page)
                
                for page in remaining_pages:
                    data += page.get('data')
                
                if not self.debug:
                    assert len(data) == pagination.get('total'), f"Pagination returned incorrect number of teams: {len(data)} / {pagination.get('total')}" 
                
        df = pd.DataFrame(data).set_index('id')

        if max_count:
            df = df.head(max_count)
        
        if attributes or members or join_link:
            team_attributes_list = []
            team_members = []
            join_links = []
            
            idx=1
            for team_id, row in df.iterrows():

                print(f"Fetching Team: {team_id} [{idx} / {len(df)}]")
                
                if members:
                    res1 = self.get_team_members(team_id)
                    if res1:
                        member_uids = set([x.get('id') for x in res1])
                    
                    res2 = self.get_team_moderators(team_id)
                    if res2:
                        # Note: This is inconsistent with the osm_id vs. just id.
                        moderator_uids = set([x.get('osm_id') for x in res2])
                                            
                    team_members.append({'id':team_id, 'member_uids':member_uids, 'moderator_uids':moderator_uids})
                    
                
                if attributes:
                    team_attributes = self.get_team_attributes(team_id)
                    if team_attributes is not None:
                        team_attributes['id'] = team_id
                        team_attributes_list.append(team_attributes)
                        
                if join_link:
                    join_link = self.get_join_link(team_id)
                    if join_link is not None:
                        #Todo: this should be sorted by created_at to get the latest join link
                        join_links.append(join_link[0])
                
                idx+=1
                
            if attributes:
                df = df.join( pd.DataFrame(team_attributes_list).set_index('id') )
                
            if members:
                df = df.join( pd.DataFrame(team_members).set_index('id') )
                
            if join_link:
                df = df.join( (pd.DataFrame(join_links).set_index('team_id')[['id']]).rename(columns={'id':'join_link'}) )

        return df
    
       
    def get_all_organization_members(
        self, 
        org_id=1, 
        org_attributes=False, 
        org_badges=False
    ):
        print("Fetching all organization members", end="")

        url = self.API_URL + 'organizations/' + str(org_id or self.ORG) + '/members'
        
        _page1 = self.session.get(url).json()
        
        if 'data' in _page1:
            data = _page1.get('data')
            
            if 'pagination' in _page1:
                pagination = _page1.get('pagination')

                print(f" - paginating {pagination.get('lastPage'):,} pages for {pagination.get('total'):,} users")
                
                remaining_pages = self._handle_pages(pagination, url)
                
                for page in remaining_pages:
                    data += page.get('data')
                
                if not self.debug:
                    assert len(data) == pagination.get('total'), "Pagination returned incorrect number of members for organization" 
    
        df = pd.DataFrame(data)
        
        df.rename(columns={'id':'uid','name':'username'}, inplace=True)
        
        df.set_index('uid', inplace=True)
        
        if org_attributes or org_badges:
            attributes = []
            badges = []
            idx = 1
            for uid, row in df.iterrows():
                print(f"Fetching UID: {uid} [{idx} / {len(df)}]")
                if org_attributes:
                    user_attributes = self.get_org_user_attributes(uid)
                    if user_attributes is not None:
                        user_attributes['uid'] = uid
                        attributes.append(user_attributes)
                if org_badges:
                    user_badges = self.get_user_badges(uid)
                    if user_badges is not None:
                        user_badges['uid'] = uid
                        badges.append(user_badges)
                        
                idx+=1
                
            if org_attributes:
                df = df.join(pd.DataFrame(attributes).set_index('uid'))
        
            if org_badges:
                df = df.join(pd.DataFrame(badges).set_index('uid'))
                        
        return df
    
    
    def get_org_user_attributes(self, uid):
        """
            Returns a dictionary of Organization attributes for a given UID
        """
        res = self.session.get(self.API_URL + f'profiles/organizations/{self.ORG}/{uid}')
        if res.text == 'Not Found' or res.status_code == 404:
            return None
        result = res.json()
        if len(result) > 0:
            try:
                return dict([(x.get('name'), x.get('value')) for x in result])
            except:
                sys.stderr.write(json.dumps(result))
                if result.get('statusCode') == 500:
                    raise Exception("API Error")
        else:
            return None
        
        
    def get_user_badges(self, uid):
        """
            Returns a dictionary of Organization badges for a given UID. 
        """
        res = self.session.get(self.API_URL + f'user/{uid}/badges')
        if res.text == 'Not Found':
            return None
        result = res.json()
        if len(result.get('badges')) > 0:
            badges = {}
            for b in result.get('badges'):
                badges[b.get('name')] = json.loads(json.dumps(b))
            return badges
        else:
            return None
        
    def get_mapper_info_from_osm(self, uids):
        count=1
        all_users = []
        this_call = []
        for uid in uids:
            if count < 50:
                this_call.append(str(uid))
                count += 1
            if count == 50:
                res = requests.get("https://openstreetmap.org/api/0.6/users.json?users=" + ",".join(this_call))
                try:
                    r = res.json()
                except:
                    print("Going to singles")
                    for uid in this_call:
                        try:
                            res = requests.get("https://openstreetmap.org/api/0.6/users.json?users=" + uid)
                            r = res.json()
                            all_users += r.get('users')
                        except:
                            print(uid + " failed")
                all_users += r.get('users')
                   
                this_call = []
                count = 1
                sys.stderr.write(f"\rUsers retrieved: {len(all_users)}"+" "*45)
        if len(this_call)>0:
            res = requests.get("https://openstreetmap.org/api/0.6/users.json?users=" + ",".join(this_call))
            all_users += res.json().get('users')
        sys.stderr.write(f"\rUsers retrieved: {len(all_users)}"+" "*45)
        
        return pd.DataFrame([r.get('user') for r in all_users]).set_index('id')

        
    
    def add_uids_to_team(self, uids, team):
        results = []
        for uid in uids:
            try:
                results.append( self.session.put(self.API_URL + f"/teams/add/{team}/{uid}") )
            except:
                print(f"Failure: {uid}")
                raise
        return results
    
    
    def update_team_location(self, team, point):
        data = json.dumps({'location': point})
        res = self.session.put(self.API_URL + f"teams/{team}", data=data)
    
    
    def add_attributes_to_team(self, team, attributes):
        curr_data = self.get_team(team)
        data = json.dumps({'hashtag':curr_data.get('hashtag'), 'tags':attributes})
        res = self.session.put(self.API_URL + f"teams/{team}", data=data)
        return res
    
    
    def assign_badge(self, org_id, badge_id, user_id, assigned_at, valid_until):
        data = {
            'assigned_at': str(assigned_at)
        }
        if valid_until:
            data['valid_until'] = str(valid_until)
        
            res = self.session.post(self.API_URL + f"organizations/{org_id}/badges/{badge_id}/assign/{user_id}",
                                    data=json.dumps(data))
        
        return res
    
    
    def get_join_link(self, team):
        res = self.session.get(self.API_URL + f"teams/{team}/invitations")
        return res.json()
    
    
    def create_org_team(self, body):
        res = self.session.post(self.API_URL + 'organizations/'+str(self.ORG)+'/teams', data=json.dumps(body))
        return res.json()
    
    
    def cache(self, df, filename):
        df.to_json(filename)
        
    
    def load(self, filename):
        pd.read_json(filename)