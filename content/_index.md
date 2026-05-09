---
# Leave the homepage title empty to use the site title
title: ''
summary: ''
date: 2022-10-24
type: landing

design:
  # Default section spacing
  spacing: '6rem'

sections:
  - block: resume-biography-3
    content:
      # Choose a user profile to display (a folder name within `content/authors/`)
      username: me
      text: ''
      # Show a call-to-action button under your biography? (optional)
      button:
        text: Curriculum Vitae (CV)
        url: uploads/resume.pdf
      headings:
        about: 'About'
        education: ''
        interests: ''
    design:
      # Use the new Gradient Mesh which automatically adapts to the selected theme colors
      background:
        gradient_mesh:
          enable: true

      # Name heading sizing to accommodate long or short names
      name:
        size: sm # Options: xs, sm, md, lg (default), xl

      # Avatar customization
      avatar:
        size: medium # Options: small (150px), medium (200px, default), large (320px), xl (400px), xxl (500px)
        shape: circle # Options: circle (default), square, rounded
  - block: markdown
    content:
      title: 'My Research'
      subtitle: ''
      text: |-
        Use this area to speak to your mission. I'm a research scientist in the Moonshot team at DeepMind. I blog about machine learning, deep learning, and moonshots.

        I apply a range of qualitative and quantitative methods to comprehensively investigate the role of science and technology in the economy.

        Please reach out to collaborate 😃
    design:
      columns: '1'
  - block: collection
    id: news
    content:
      title: News
      subtitle: ''
      text: ''
      # Choose how many pages you would like to display (0 = all pages)
      count: 5
      # Page type to display. E.g. post, talk, publication...
      page_type: blog
      # Filter on criteria
      filters:
        author: ''
        category: 'news'
        tag: ''
        exclude_featured: false
        exclude_future: false
        exclude_past: false
      # Choose how many pages you would like to offset by
      offset: 0
      # Page order: descending (desc) or ascending (asc) date.
      order: desc
    design:
      # Choose a layout view
      view: date-title-summary
  - block: collection
    id: publications
    content:
      title: Featured Publications
      count: 6
      filters:
        folders:
          - publications
        featured_only: true
    design:
      view: article-grid
      columns: 2
  - block: collection
    id: posts
    content:
      title: Recent Posts
      subtitle: ''
      text: ''
      # Choose how many pages you would like to display (0 = all pages)
      count: 8
      # Page type to display. E.g. post, talk, publication...
      page_type: blog
      # Filter on criteria
      filters:
        author: ''
        category: 'technical'
        tag: ''
        exclude_featured: false
        exclude_future: false
        exclude_past: false
      # Choose how many pages you would like to offset by
      offset: 0
      # Page order: descending (desc) or ascending (asc) date.
      order: desc
    design:
      # Choose a layout view
      view: article-grid
      columns: 2
  - block: collection
    id: talks
    content:
      title: Recent & Upcoming Talks
      filters:
        folders:
          - events
    design:
      view: compact
  - block: resume-experience
    id: experience
    content:
      title: ''
      subtitle: ''
      username: me
    design:
      date_format: 'January 2006'
      is_education_first: false
  - block: collection
    content:
      title: Publications
      text: ''
      count: 0
      filters:
        folders:
          - publications
        exclude_featured: false
    design:
      view: citation
  - block: contact-info
    id: contact
    content:
      title: Contact
      subtitle: ''
      text: Drop me a line
      email: louis@tiao.io
      autolink: true
---
