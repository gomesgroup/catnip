import React, {useRef, useState} from 'react';
import {useNavigate} from 'react-router-dom';
import KetcherExample from '../sketcher_component.tsx';
import {Typography, Button, Box, Grid, Container, Link, TextField, Tab, Tabs, Paper} from '@mui/material';
import AppBarComponent from '../components/AppBarComponent';
import ScatterPlot from "../components/ScatterPlot";
import {data} from "../data";
import { GitHub, Code } from '@mui/icons-material';

const TabPanel = (props) => {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
};

const a11yProps = (index) => {
  return {
    id: `simple-tab-${index}`,
    'aria-controls': `simple-tabpanel-${index}`,
  };
};

function SubmitPage() {
  const ketcherRef = useRef(null);
  const compatibilityKetcherRef = useRef(null);
  const navigate = useNavigate();
  const [tabValue, setTabValue] = useState(0);
  const [enzymeSequence, setEnzymeSequence] = useState('');
  const [compatibilityEnzymeSequence, setCompatibilityEnzymeSequence] = useState('');

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handleSubmission = async () => {
    if (tabValue === 0 && ketcherRef.current) {
      try {
        const moleculeData = await ketcherRef.current.getMoleculeData();

        const response = await fetch('/api/submit', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({data: moleculeData}),
        });

        const result = await response.json();
        const runId = result.run_id;

        navigate(`/run/${runId}`);

      } catch (error) {
        console.error('Failed to fetch molecule data:', error);
      }
    } else if (tabValue === 1 && enzymeSequence.trim()) {
      try {
        const response = await fetch('/api/submit-enzyme', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({sequence: enzymeSequence}),
        });

        const result = await response.json();
        const runId = result.run_id;

        navigate(`/enzyme-run/${runId}`);
      } catch (error) {
        console.error('Failed to submit enzyme sequence:', error);
      }
    } else if (tabValue === 2 && compatibilityKetcherRef.current && compatibilityEnzymeSequence.trim()) {
      try {
        const moleculeData = await compatibilityKetcherRef.current.getMoleculeData();
        
        const response = await fetch('/api/calculate-compatibility', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            substrate: moleculeData,
            sequence: compatibilityEnzymeSequence
          }),
        });

        const result = await response.json();
        const runId = result.run_id;

        navigate(`/compatibility/${runId}`);
      } catch (error) {
        console.error('Failed to calculate compatibility:', error);
      }
    }
  };

  return (
    <div>
      <AppBarComponent/>
      <Container maxWidth="xl">
        <Grid container spacing={8}>
          <Grid item xs={6}>
            <Typography variant="h4" sx={{marginTop: "128px", marginBottom: '64px'}}>
              Generation of connections between protein sequence space and chemical space to enable a predictive model for biocatalysis
            </Typography>
            <Typography variant="body1">
The application of biocatalysis in synthesis has the potential to offer dramatically streamlined routes toward target molecules, exquisite and tunable catalyst-controlled selectivity, as well as more sustainable processes. Despite these advantages, biocatalytic synthetic strategies can be high risk to implement. Successful execution of these approaches requires identifying an enzyme capable of performing chemistry on a specific intermediate in a synthesis which often calls for extensive screening of enzymes and protein engineering. Strategies for predicting which enzyme is most likely to be compatible with a given small molecule have been hindered by the lack of well-studied biocatalytic reactions. The under exploration of connections between chemical and protein sequence spaces constrains navigation between these two landscapes. Herein, this longstanding challenge is overcome in a two-phase effort relying on high throughput experimentation to populate connections between substrate chemical space and biocatalyst sequence space, and the subsequent development of machine learning models that enable the navigation between these two landscapes. Using a curated library of α-ketoglutarate-dependent non-heme iron (NHI) enzymes, the <code>BioCatSet1</code> dataset was generated to capture the reactivity of each biocatalyst with {">"} 100 substrates. In addition to the discovery of novel chemistry, <code>BioCatSet1</code> was leveraged to develop a predictive workflow that provides a ranked list of enzymes that have the greatest compatibility with a given substrate. To make this tool accessible to the community, we built <b>CATNIP</b>, an open access web interface to our predictive workflows. We anticipate our approach can be readily expanded to additional enzyme and transformation classes, and will derisk the application of biocatalysis in chemical synthesis.
            </Typography>

          </Grid>

          <Grid item xs={6} style={{textAlign: 'left'}}>
            <Paper sx={{ marginTop: "128px" }}>
              <Tabs value={tabValue} onChange={handleTabChange} aria-label="input tabs">
                <Tab label="Substrate to Enzyme" {...a11yProps(0)} />
                <Tab label="Enzyme to Substrate" {...a11yProps(1)} />
              </Tabs>

              <TabPanel value={tabValue} index={0}>
                <Typography variant={"h6"} sx={{ marginBottom: 3 }}>
                  Draw your substrate
                </Typography>
                <Box border="1px solid lightgray">
                  <KetcherExample ref={ketcherRef}/>
                </Box>
              </TabPanel>

              <TabPanel value={tabValue} index={1}>
                <Typography variant={"h6"} sx={{ marginBottom: 3 }}>
                  Enter your enzyme sequence
                </Typography>
                <TextField
                  fullWidth
                  multiline
                  rows={8}
                  placeholder="Paste your enzyme sequence here..."
                  value={enzymeSequence}
                  onChange={(e) => setEnzymeSequence(e.target.value)}
                  variant="outlined"
                />
              </TabPanel>

              <Box p={3}>
                <Button variant="contained" color="primary" onClick={handleSubmission} sx={{backgroundColor: "#C41230"}}>
                  Submit
                </Button>
              </Box>
            </Paper>
          </Grid>
        </Grid>
      </Container>
        <Box sx={{backgroundColor: "#C41230", marginTop: "64px", paddingTop: "2px", paddingBottom:"2px"}}>
            <Container maxWidth="xl">
                <Grid container spacing={8}>
                    <Grid item xs={12} sx={{color: "white"}}>
                        <p>Raw data is available at <Button variant="contained"  href="https://huggingface.co/gomesgroup/catnip/tree/main/raw_data" target="_blank" sx={{marginLeft: "8px", textTransform:"none"}}>Hugging Face</Button></p>
                    </Grid>
                </Grid>
            </Container>
        </Box>
      <Box sx={{backgroundColor: "black", paddingTop: "64px", paddingBottom:"64px"}}>
        <Container maxWidth="xl">
          <Grid container spacing={8}>
              <Grid item xs={6}>
              <Typography variant="h6" sx={{color: "white", marginBottom: "48px"}}>
            ... or explore already existing space
              </Typography>
              <Box sx={{
                width: '100%', // or any width you prefer
                height: 0,
                paddingBottom: '100%',
                position: 'relative'
              }}>
                <div style={{position: 'absolute', top: 0, left: 0, right: 0, bottom: 0}}>
                  <ScatterPlot data={data}/>
                </div>
              </Box>
              <br/>
            </Grid>
              <Grid item xs={6}>
              <Box sx={{
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                height: '100%' // This ensures the Box takes full height of its container
              }}>
                <Box sx={{
                  display: 'flex',
                  justifyContent: 'center',
                  gap: 4
                }}>
                  <Link
                    href="https://github.com/gomesgroup/catnip"
                    target="_blank"
                    rel="noopener noreferrer"
                    sx={{
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      color: 'white',
                      textDecoration: 'none',
                      '&:hover': { color: '#C41230' }
                    }}
                  >
                    <GitHub sx={{ fontSize: 48, marginBottom: 1 }} />
                    <Typography variant="subtitle1">GitHub</Typography>
                  </Link>
                  <Link
                    href="https://huggingface.co/gomesgroup/catnip"
                    target="_blank"
                    rel="noopener noreferrer"
                    sx={{
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      color: 'white',
                      textDecoration: 'none',
                      '&:hover': { color: '#C41230' }
                    }}
                  >
                    <Code sx={{ fontSize: 48, marginBottom: 1 }} />
                    <Typography variant="subtitle1">Hugging Face</Typography>
                  </Link>
                </Box>
              </Box>

            </Grid>
          </Grid>
        </Container>
      </Box>
    </div>
  );
}

export default SubmitPage;
