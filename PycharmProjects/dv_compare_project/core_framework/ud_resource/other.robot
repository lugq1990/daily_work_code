*** Settings ***
Documentation     A resource file with reusable keywords and variables.
...
...               The system specific keywords created here form our own
...               domain specific language. They utilize keywords provided
...               by the imported SeleniumLibrary.
Resource          seleow.robot
Resource          resource.robot
Library           SeleniumLibraryExtension    run_on_failure=Nothing
Library           StringLibraryExtension


*** Keywords ***

Get expect data from lake
    [Arguments]    ${sql}    ${schema}
    Set Test Variable    ${query}    ${sql}
    Set Test Variable    ${data_schema}    ${schema}

Get Expect Result From
    [Arguments]    ${p}
    Sleep    1 second

Set Filters
    [Arguments]    ${p}
    Sleep    1 second

Set Object Id
    [Arguments]    ${objid}
    Set Test Variable    ${objid}    ${objid}

Load Shared Parameter
    Sleep    1 second

Compare actual and expect data
    Compare Data    csv_has_header=True

Compare actual with no header and expect data
    Compare Data    csv_has_header=False

Compare Data
    [Arguments]    ${csv_has_header}
    # Wait Until Keyword Succeeds    3x    3seconds    SeleniumLibraryExtension.Compare with database data    has_header=${csv_has_header}
    SeleniumLibraryExtension.Compare with database data    has_header=${csv_has_header}

Get Actual and Expect data and Compare them
    [Arguments]    ${params}
    [Documentation]
    @{param_list} =    Split String    ${params}    `$`
    ${objid}    Get From List    ${param_list}    0
    ${filters}    Get From List    ${param_list}    1
    ${query}    Get From List    ${param_list}    2
    ${schema}    Get From List    ${param_list}    3
    Set Test Variable    ${query}    ${query}
    Set Test Variable    ${schema}    ${schema}
    # Switch Active Browser    https://bi8.ciostage.accenture.com/azure/single/?appid=83a1e58c-08f8-4fbe-9951-e7dc4c60b5bb
    ${single_url} =    Get single obj url    ${dashboard_url}    ${objid}    ${filters}
    log        ${single_url}
    Execute Javascript    window.open()
    Switch Window    locator=NEW
    ${status}	${value}    Run Keyword And Ignore Error  Go To    ${single_url}
    Run Keyword If	'${status}' == 'FAIL'  Close Browser
    Run Keyword If	'${status}' == 'FAIL'  Open Browser And Login Dashboard  ${single_url}
    Right Click and Export Data For QlikSense    class:qv-object-content-container    xpath://span[@title="Export data"]  class:export-url
    Compare actual and expect data