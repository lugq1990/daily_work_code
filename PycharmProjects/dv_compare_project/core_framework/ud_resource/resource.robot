*** Settings ***
Documentation     A resource file with reusable keywords and variables.
...
...               The system specific keywords created here form our own
...               domain specific language. They utilize keywords provided
...               by the imported SeleniumLibrary.
Resource          seleow.robot
Library           SeleniumLibraryExtension    run_on_failure=Nothing
Library           OperatingSystem
Library           Collections

*** Variables ***
${BROWSER}        Chrome
${DELAY}          0

# ${options}    add_experimental_option('excludeSwitches', ['enable-logging']);add_argument('--disable-dev-shm-usage');add_argument('--no-sandbox');add_argument('--headless')
# ${options}    add_experimental_option('excludeSwitches', ['enable-logging'])
${TIME OUT}    60 seconds
*** Keywords ***

BI7 Dashboard Login
    [Arguments]    ${dashboard_url}
    Open Dashboard In Browser
    # Add Cookie    AccessPointSession    5a11880d-91af-f561-4fd4-465ac5268c23    domain=bi7-azure.ciostage.accenture.com

    #Add Cookie    Qlik-KEY    report=%2fQvAJAXZfc%2fopendoc.htm%3fdocument%3dalice_reports%255Ctraining.qvw&counter=1    domain=bi7-azure.ciostage.accenture.com
    #Add Cookie    qlikmachineid    7f94f568-4c75-4146-9830-b86c9f2bc3dd    domain=bi7-azure.ciostage.accenture.com
    #Add Cookie    AWSELB    95A3E1C5083E7B9203CD947DD742CE19B1CB514A0831CE5C31FC4359FAEF7DE2193465F7F313FDC9D3DF8A9AE7FF03824206256AA6842135F29E0ACDC5E3AFA10B9586885F    domain=bi7-azure.ciostage.accenture.com
    #Add Cookie    AWSELBCORS    95A3E1C5083E7B9203CD947DD742CE19B1CB514A0831CE5C31FC4359FAEF7DE2193465F7F313FDC9D3DF8A9AE7FF03824206256AA6842135F29E0ACDC5E3AFA10B9586885F	    domain=bi7-azure.ciostage.accenture.com
    Wait For Condition    return document.readyState=="complete"
    # Set Cookie    AccessPointSession    5a11880d-91af-f561-4fd4-465ac5268c23    domain=bi7-azure.ciostage.accenture.com

Set dashboard url
    [Arguments]    ${url}
    Set Test Variable    ${dashboard_url}    ${url}

Open Dashboard In Browser
    # Create Directory    ${download directory}
    # Wait Until Keyword Succeeds    3x    3seconds    Open Browser    ${dashboard_url}    ${BROWSER}    options=${options}    executable_path=${executable_path}
    Switch Or Open Browser For Dashboard
    # Open Browser    about:blank    ${BROWSER}    alias=myChrome    options=${options}    executable_path=${executable_path}
    # Maximize Browser Window
    Set Selenium Speed    ${DELAY}
    # Login Page Should Be Open


Switch Or Open Browser For Dashboard
    Set Common Vars
    SET CHROME DOWNLOAD FOLDER    ${download directory}
    ${alias}    Get Browser Aliases
    ${status}    ${value}    Run Keyword And Ignore Error    Dictionary Should Contain Key    ${alias}    myChrome
    Run Keyword If	'${status}' == 'PASS'	Switch Active Browser    ${dashboard_url}
    Run Keyword If	'${status}' == 'FAIL'	Open Browser And Login Dashboard
    Log    ${alias}

Switch Active Browser
    [Arguments]    ${dashboard_url}
    Switch Browser    myChrome
    # Switch Window    locator=main
    # Open New Tab On Browser
    #Sleep    5 seconds
    ${status}	${value}    Run Keyword And Ignore Error  Go To    ${dashboard_url}
    Run Keyword If	'${status}' == 'FAIL'  Close Browser
    Run Keyword If	'${status}' == 'FAIL'  Open Browser And Login Dashboard

Open Browser And Login Dashboard
    [Arguments]    ${_url}=None
    ${executable_path}    Get Executable Path
    ${options}    GET OPTIONS
    Set Suite Variable    ${chrome_options}    ${options}
    Open Browser    ${dashboard_url}    ${BROWSER}    alias=myChrome    options=${chrome_options}    executable_path=${executable_path}
    Maximize Browser Window
    Input Username    ${useremail}
    Submit Login User
    Input Password    ${password}
    Submit Credentials
    Sleep    5 seconds
    Run Keyword If  ${_url} is not None  Go To    ${_url}

Login Page Should Be Open
    Title Should Be    QlikView

Go To Login Page
    [Arguments]    ${dashboard_url}
    Go To    ${dashboard_url}
    Login Page Should Be Open

Input Username
    [Arguments]    ${username}
    Wait Until Element Is Visible    id:i0116    ${TIME OUT}
    Input Text    id:i0116    ${username}

Input Password
    [Arguments]    ${password}
    Wait Until Element Is Visible    name:Password    ${TIME OUT}
    Input Text    name:Password    ${password}

Submit Login User
    Click Button    xpath://*[@type="submit"]

Submit Credentials
    Press Keys    submitButton    ENTER

Set Common Vars
    ${time now}    Get Time    epoch
    ${time now}    Convert To String    ${time now}
    ${download_directory}    Join Path    ${OUTPUT DIR}    downloads    ${TEST NAME}    ${time now}
    Set Test Variable    ${download directory}
    # [Return]    ${download directory}

