# ELS-EEW: Dataset Indices and Data Acquisition Guide

## Overview

This archive provides research materials for identifying and reconstructing the earthquake waveform dataset used in the **ELS-EEW (Edge-oriented Lightweight Streaming Earthquake Early Warning)** study. It is intended to support transparent dataset selection and reproducible experiments without redistributing third-party strong-motion waveform files.

The seismic observations originate from the **K-NET and/or KiK-net strong-motion networks**, operated by the **National Research Institute for Earth Science and Disaster Resilience (NIED), Japan**. These records must be obtained directly from NIED by each user, subject to NIED's registration procedure and data-use conditions.

**No NIED waveform files are included in this archive.** The accompanying record indices and dataset-construction materials are intended to help researchers identify the records used in the study and reproduce the authors' dataset-building procedure, to the extent supported by the released materials.

## 1. Original data source and citation

**Provider:** National Research Institute for Earth Science and Disaster Resilience (NIED)  
**Networks:** K-NET (Kyoshin Network) and KiK-net (Kiban Kyoshin Network)  
**Official data portal:** https://www.kyoshin.bosai.go.jp/en/  
**Original dataset DOI:** https://doi.org/10.17598/NIED.0004

Please cite the original NIED dataset whenever you use these records:

> National Research Institute for Earth Science and Disaster Resilience. (2019). *NIED K-NET, KiK-net* [Dataset]. National Research Institute for Earth Science and Disaster Resilience. https://doi.org/10.17598/NIED.0004

The DOI identifies the **original NIED dataset**, not the authors' record-index archive.

## 2. How to obtain the Japanese strong-motion data

### Step 1: Register for NIED data access

NIED requires user registration before downloading K-NET and KiK-net waveform data.

1. Open the official K-NET/KiK-net portal: https://www.kyoshin.bosai.go.jp/en/
2. Select **User Registration** / **New User Registration**.
3. Read NIED's data-use instructions and proceed through the registration form: https://hinetwww11.bosai.go.jp/nied/registration/
4. Complete the registration procedure and obtain the account credentials required for data downloads.
5. Follow any further instructions shown by NIED. Account registration and access procedures may change.

Registration and account questions should be directed to NIED through the contact information on its official [FAQ page](https://www.kyoshin.bosai.go.jp/en/faq/).

### Step 2: Locate earthquake records

NIED provides several official ways to find and download waveforms:

**Option A — Search by earthquake (recommended for specific events):**  
https://www.kyoshin.bosai.go.jp/en/eqdownload/

- Search for an earthquake using its date and other available event information.
- Select the matching event and check the **Strong-Motion Record List**.
- Match the station code and network against the record indices supplied with this archive.
- Choose a waveform format (for example, **K-NET ASCII**, **K-NET Binary**, or **K-NET CSV**) and download the records through NIED's authenticated service.

**Option B — Search by data conditions (recommended for station- or criteria-based searches):**  
https://www.kyoshin.bosai.go.jp/en/dtdownload/

- Search using relevant conditions such as date, network, station code, magnitude, and epicentral distance.
- Compare the search results with the archived record indices.
- Download the corresponding waveforms through NIED.

**Option C — HTTPS download (recommended for bulk access):**  
https://www.kyoshin.bosai.go.jp/en/https_download/

- Sign in using your own registered NIED credentials.
- Consult NIED's official documentation for the directory layout and authenticated `wget` examples.
- Select only the event/station records needed to reconstruct the dataset.
- Do **not** put your credentials into public scripts, repositories, logs, or shared files.

The official HTTPS repository is at:

https://www.kyoshin.bosai.go.jp/kyoshin/download/

For example, NIED documents the following **directory patterns** (these are directory patterns, not a list of records used in ELS-EEW):

```text
knet/data/YYYY/MM/YYYYMMDDHHMISS/
kik/data/YYYY/MM/YYYYMMDDHHMISS/
knet/zip/YYYY/MM/YYYYMMDDHHMISS/
kik/zip/YYYY/MM/YYYYMMDDHHMISS/
```

Here, `YYYYMMDDHHMISS` is the event directory identifier used by NIED. Confirm it against the portal's event record rather than relying solely on a rounded origin-time value or an independently assigned event ID.

### Step 3: Identify the correct waveform components

For **K-NET**, three-component strong-motion records contain:

- `NS`: north–south acceleration;
- `EW`: east–west acceleration;
- `UD`: vertical (up–down) acceleration.

For **KiK-net**, instruments may be installed both at the surface and in a borehole. In K-NET ASCII-format filenames, NIED identifies:

- `NS1`, `EW1`, `UD1`: borehole components;
- `NS2`, `EW2`, `UD2`: surface components.

Choose the correct installation level when matching the study's record indices. Do not substitute borehole waveforms for surface waveforms (or vice versa) unless the study's data-construction specification explicitly requires it.

For K-NET ASCII files, the waveform sample values are digital counts and the header's **Scale Factor** is needed to convert them into physical acceleration units. NIED's CSV format provides physical values. Preserve the original sampling rate and metadata, and follow the conversion and preprocessing procedures implemented in the accompanying study code.

See the official [NIED FAQ](https://www.kyoshin.bosai.go.jp/en/faq/) for file components, formats, and scaling conventions.

## 3. Using the archived record indices

The record indices supplied in this archive identify the observations selected for the study. Use the **actual columns and identifiers present in the supplied index files**; the following are general matching concepts rather than a claim about any particular filename or schema:

- Earthquake/event identifier or event directory time;
- Observation network (K-NET or KiK-net);
- Station code and, where relevant, sensor installation level;
- Recording time and/or the specific waveform file identifier;
- Dataset assignment (training, validation, or test), if included;
- P-wave arrival information and any study-specific selection metadata, if included.

**Important:** NIED states that the origin-time seconds shown in some event metadata may be placeholders (such as `00` or `30`). Match records using the NIED event listing and station/file identifiers rather than assuming that these seconds represent a precise physical origin time. NIED does **not** provide P- and S-wave arrival picks in its waveform download service; any picks distributed in this archive are study-derived information, not official NIED picks.

If a particular index row cannot be matched unambiguously to a NIED record, consult the accompanying dataset documentation rather than substituting a different event or station.

## 4. Reconstructing the research dataset

After downloading the appropriate waveforms from NIED:

1. **Preserve original files.** Keep the NIED-downloaded waveforms and their headers unchanged in a local, access-controlled directory.
2. **Match records.** Use the archived indices to identify the corresponding events, stations, channels, and recording files.
3. **Apply dataset construction.** Run the dataset-selection, unit-conversion, waveform-windowing, and preprocessing code supplied with the ELS-EEW project, using the configuration corresponding to the study.
4. **Apply the supplied partitions.** Reproduce the training, validation, and test assignments from the archived index information rather than generating a new random partition.
5. **Verify outputs.** Compare record counts, sample identifiers, waveform lengths, and other documented checks against the dataset description in the associated research work.

The ELS-EEW source-code repository is:

https://github.com/BeiFangDeMao/ELS-EEW

For a research reproduction, use the **specific archived software release cited by the associated paper**, rather than assuming that the latest `main` branch reproduces historical experiments. Any additional dependencies or commands should be taken from the released source-code documentation; this README does not prescribe unverified script names or command-line arguments.

## 5. Data access, ownership, and redistribution

The original K-NET/KiK-net waveforms are third-party data provided by NIED. NIED's official website and registration agreement restrict duplication and redistribution of its files. Accordingly:

- This archive **does not mirror or redistribute original NIED waveform files**.
- This archive **does not provide repackaged or preprocessed NIED waveform files**.
- Researchers wishing to use the seismic records must access them independently through the **official NIED services**, accept the applicable conditions, and cite the original NIED dataset.
- The licensing or access terms attached to the authors' own indices, documentation, or code **do not grant rights to redistribute NIED waveform data**.

The authors' record indices and reconstruction materials should be used only in accordance with the applicable rights and conditions governing those materials. If exact reuse or redistribution permissions are uncertain, consult NIED before republishing third-party content.

NIED's data-use notice is available at https://www.kyoshin.bosai.go.jp/en/ and its registration agreement at https://hinetwww11.bosai.go.jp/nied/registration/.

## 6. Citing this project

If you use the dataset catalogs, labels, source code, preprocessing scripts, or model weights provided by this project, please cite **this project's DOI**. These materials form a single ELS-EEW archive and share the same project DOI; separate dataset-catalog and model-software DOI citations are not required.

When using the original NIED waveforms, also cite the original NIED dataset independently:

> National Research Institute for Earth Science and Disaster Resilience. (2019). *NIED K-NET, KiK-net* [Dataset]. National Research Institute for Earth Science and Disaster Resilience. https://doi.org/10.17598/NIED.0004

The project DOI and the NIED dataset DOI identify different resources and do not replace one another.

The four-column catalogs contain record identifiers, magnitude, epicentral distance, and back-azimuth labels, but not manually annotated P-wave arrival times. The causal processing script assumes P-aligned inputs; exact reconstruction of the study's input windows still requires the corresponding arrival annotations.

## 7. Official resources

- K-NET/KiK-net home and data-use policy: https://www.kyoshin.bosai.go.jp/en/
- NIED user registration: https://hinetwww11.bosai.go.jp/nied/registration/
- Download by earthquake: https://www.kyoshin.bosai.go.jp/en/eqdownload/
- Download by data condition: https://www.kyoshin.bosai.go.jp/en/dtdownload/
- Download by HTTPS and directory specifications: https://www.kyoshin.bosai.go.jp/en/https_download/
- Data formats, component definitions, and FAQ: https://www.kyoshin.bosai.go.jp/en/faq/
- NIED K-NET/KiK-net dataset citation: https://doi.org/10.17598/NIED.0004

---

**Scope note:** This README documents how to access NIED source observations and how the publicly released study materials are intended to support dataset reconstruction. It does not imply that the third-party waveform files themselves are openly redistributable, or that any index entry can be retrieved without complying with NIED's access conditions.
