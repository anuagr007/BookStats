
import streamlit as st
import requests
import json
import re
import pandas as pd
import time
import numpy as np
import regex
import extra_streamlit_components as stx
from streamlit_pills import pills
import pandas as pd
from bs4 import BeautifulSoup
import urllib.parse
from urllib.request import urlopen
import spacy
from streamlit_option_menu import option_menu
import random
import cv2
from PIL import ImageFont, ImageDraw, Image
from typing import Optional, Tuple
from typing import Literal
from streamlit_mic_recorder import mic_recorder,speech_to_text

# Page setting
st.set_page_config(layout="wide")



css='''

    <style>

        section.main>div {

            padding-bottom: 1rem;

        }

       div[data-testid="column"]:has(div[class="scrollable-col4"]){

            overflow: scroll;

        }

    </style>

    '''


st.markdown(css, unsafe_allow_html=True)

st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

def header(url):
     st.markdown(f'<p style="font-size:40px;border-radius:1%;text-align:center;font-weight: bold;">{url}</p>', unsafe_allow_html=True)

def add_text_to_image(
    image_rgb: np.ndarray,
    label: str,
    top_left_xy: Tuple = (0, 0),
    font_scale: float = 1,
    font_thickness: float = 1,
    font_face=cv2.FONT_HERSHEY_DUPLEX,
    font_color_rgb: Tuple = (0, 0, 255),
    bg_color_rgb: Optional[Tuple] = None,
    outline_color_rgb: Optional[Tuple] = None,
    line_spacing: float = 1,
):
    """
    Adds text (including multi line text) to images.
    You can also control background color, outline color, and line spacing.

    outline color and line spacing adopted from: https://gist.github.com/EricCousineau-TRI/596f04c83da9b82d0389d3ea1d782592
    """
    OUTLINE_FONT_THICKNESS = 3 * font_thickness

    im_h, im_w = image_rgb.shape[:2]

    for line in label.splitlines():
        x, y = top_left_xy

        # ====== get text size
        if outline_color_rgb is None:
            get_text_size_font_thickness = font_thickness
        else:
            get_text_size_font_thickness = OUTLINE_FONT_THICKNESS

        (line_width, line_height_no_baseline), baseline = cv2.getTextSize(
            line,
            font_face,
            font_scale,
            get_text_size_font_thickness,
        )
        line_height = line_height_no_baseline + baseline

        if bg_color_rgb is not None and line:
            # === get actual mask sizes with regard to image crop
            if im_h - (y + line_height) <= 0:
                sz_h = max(im_h - y, 0)
            else:
                sz_h = line_height

            if im_w - (x + line_width) <= 0:
                sz_w = max(im_w - x, 0)
            else:
                sz_w = line_width

            # ==== add mask to image
            if sz_h > 0 and sz_w > 0:
                bg_mask = np.zeros((sz_h, sz_w, 3), np.uint8)
                bg_mask[:, :] = np.array(bg_color_rgb)
                image_rgb[
                    y : y + sz_h,
                    x : x + sz_w,
                ] = bg_mask

        # === add outline text to image
        if outline_color_rgb is not None:
            image_rgb = cv2.putText(
                image_rgb,
                line,
                (x, y + line_height_no_baseline),  # putText start bottom-left
                font_face,
                font_scale,
                outline_color_rgb,
                OUTLINE_FONT_THICKNESS,
                cv2.LINE_AA,
            )
        # === add text to image
        image_rgb = cv2.putText(
            image_rgb,
            line,
            (x, y + line_height_no_baseline),  # putText start bottom-left
            font_face,
            font_scale,
            font_color_rgb,
            font_thickness,
            cv2.LINE_AA,
        )
        top_left_xy = (x, y + int(line_height * line_spacing))

    return image_rgb

def book_info(bookname):
    base_url = 'https://www.goodreads.com/search?'
    params = {'q': bookname}
    search_url = base_url + urllib.parse.urlencode(params)
    search_page = urlopen(search_url)
    search_html = search_page.read().decode("utf-8")
    search_soup = BeautifulSoup(search_html, "html.parser")
    links_with_text = []
    for a in search_soup.find_all('a', href=True): 
        if a.text: 
            links_with_text.append(a['href'])
    matching = [s for s in links_with_text if "/book/show/" in s][0]
    book_name = matching.split('?')[0]
    complete_url = "https://www.goodreads.com" + book_name
    #url = url2
    page = urlopen(complete_url)
    html = page.read().decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string
    title = title.split('|')[0].strip()
    rating = soup.find_all("div", {"class": "RatingStatistics__rating"})[0].string
    rating_cnt = soup.find("span", {"data-testid": "ratingsCount"}).get_text()
    desc = soup.find("div", {"class": "BookPageMetadataSection__description"}).get_text()
    genre = soup.find_all("div", {"class": "BookPageMetadataSection__genres"})
    images = soup.find_all("img", {"class": "ResponsiveImage"})[0]
    img_src = images['src']
    genres = []
    if len(genre) == 0:
        genres = ["Good to read","Fun"]
    else:
        for item in genre:
            x = item.find_all('span', attrs={'class':'Button__labelItem'})

        array_length = len(x)
        for item in range(array_length):
            b = str(x[item])
            start = b.find('>') + 1
            end = b.find('</')
            genres.append(b[start:end])

    #genres = genres[:-1]
    time.sleep(1)
    return title, rating, desc, genres,rating_cnt,img_src
    
def create_img(book_title,user_input):
    book_title1 = book_title.split("by")
    
    words = user_input.split()
    grouped_words = [' '.join(words[i: i + 2]) for i in range(0, len(words), 2)]

    words1 = book_title1[1].split()
    grouped_words1 = [' '.join(words1[i: i + 2]) for i in range(0, len(words1), 2)]

    grouped_words.insert(len(grouped_words),"    By     ")

    final = grouped_words + grouped_words1

    i = 1
    while i < len(final):
        final.insert(i, '\n')
        i += 2
    
    img_path = ["m1.jpg","m4.jpg","m7.jpg"]
 
    for i in img_path:
        image = cv2.imread(i)

        height = np.size(image, 0)
        width = np.size(image, 1)

        (h, w) = image.shape[:2]


        cv2.circle(image, (w//2, h//2), 145, (0, 0, 0), -1)

        text_til =  " ".join(final)


        image = add_text_to_image(
            image,
            text_til,
            font_color_rgb=(255, 255, 255),
            outline_color_rgb=(0, 0, 0),
            top_left_xy=(w//2 - 82, h//2 -82),
            line_spacing=1.5,
            font_face=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            font_scale = 0.7
        )

        cv2.imwrite("poster_" + i[0:2] + ".png", image)
        
    img_path1 = ["m2.jpg","m3.jpg"]

    for i in img_path1:
        image1 = cv2.imread(i)

        height1 = np.size(image1, 0)
        width1 = np.size(image1, 1)

        (h, w) = image1.shape[:2]


        cv2.circle(image1, (w//2, h//2), 120, (0, 0, 0), -1)

        text_til =  " ".join(final)


        image1 = add_text_to_image(
                                    image1,
                                    text_til,
                                    font_color_rgb=(255, 255, 255),
                                    outline_color_rgb=(0, 0, 0),
                                    top_left_xy=(w//2 - 72, h//2 -82),
                                    line_spacing=1.5,
                                    font_face=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                                    font_scale = 0.7
                                )

        cv2.imwrite("poster_" + i[0:2] + ".png", image1)

 
        img_path2 = r"m6.jpg"

        image2 = cv2.imread(img_path2)

        (h, w) = image2.shape[:2]


        cv2.circle(image2, (w//2, h//2-15), 200, (0, 0, 0), -1)

        text_til =  " ".join(final)


        image2 = add_text_to_image(
            image2,
            text_til,
            font_color_rgb=(255, 255, 255),
            outline_color_rgb=(0, 0, 0),
            top_left_xy=(w//2 - 72, h//2 -82),
            line_spacing=1.5,
            font_face=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            font_scale = 0.7
        )

        cv2.imwrite("poster_m6.png", image2)


    return True

def select_image():
    imgs = ["poster_m1.png","poster_m2.png","poster_m3.png","poster_m4.png","poster_m6.png","poster_m7.png"]
    img_list = random.sample(imgs, 3)
    return img_list


header("📚 Know your Books Better!")

con2 = st.container(border = True)
with con2:
    v1,v2 = st.columns((0.6,4))
    with v1:
        st.markdown("**Hello Bibliophile 🙇‍♂️**") 
    with v2:
        
        selected2 = option_menu(None, ["Search","Write"], 
            icons=['search','pen'], 
            menu_icon="cast", default_index=0, orientation="horizontal")

if selected2 == "Write":
    
    con4 = st.container(border = True)
    
    with con4:
        st.header("Your Personal Book Review Blog")
        st.warning("Save your thoughts about the books you read")
        
        rev = st.text_area('(Press Ctrl+Enter to save and then Download)', value = "Start Jotting down your review")
        #img = st.file_uploader("Upload the images you relate with", type=['jpeg','png','jpg'])
        st.download_button("⬇ Download",rev)
            
            
                

if selected2 == "Search":

    prompt_placeholder = st.form("chat-form")

    with prompt_placeholder:
        
        cols = st.columns((6,1.5))
        with cols[0]:
            try:
                user_input = st.text_input("ERT",
                placeholder="Which book is on your mind today?",
                label_visibility="collapsed")
            except:
                st.info("Enter book of your choice")
        
        with cols[1]:
            frm_smt = st.form_submit_button("Submit")


    con1 = st.container(border = True)
    
    con3 = st.container(border = True)

    try:
        book_des = book_info(user_input)
        book_title = book_des[0]
        book_summary = book_des[2]
        book_rating = book_des[1]
        book_genre = book_des[3]
        rating_count = book_des[4]
        book_image = book_des[5]
        
            
        if "...more" in book_genre:
            book_genre.remove("...more")

        if frm_smt:
            with con1:
                c1,c2 = st.columns((1,2.5))
                with c2:
                    st.subheader("📑 :blue[Synopsis] ")
                    st.info(book_title)
                    st.write(book_summary)
                with c1:
                    st.image(book_image, width = 300)
                    
    
    
           
    
    
            with con3:
                c6,c7 = st.columns(2)
    
                with c7:
                    st.subheader("🖌️ :blue[Custom Cover] ")
                    create_img(book_title,user_input)
                    img_list = select_image()
                    v1,v2,v3 = st.columns(3)
                    with v1:
                        st.image(img_list[0])
                    with v2:
                        st.image(img_list[1])
                    with v3:
                        st.image(img_list[2])
                        
                with c6:
                    g2 = st.container()
                    g3 = st.container()
                    g4 = st.container()
                    
                    with g2:
                        st.subheader("🎭 :blue[Genre]")
                        pills('This book is',book_genre, key = "p2",index = None)
                    with g3:
                        st.subheader("👍 :blue[User Rating] ")
                        coo1,coo2 = st.columns((1,8))
                        with coo1:
                            st.header(book_rating)
                        with coo2:
                            st.write(rating_count,)
                        if float(book_rating) > 4.00:
                            st.write(":star2: :star2: :star2: :star2:")
                        elif float(book_rating) > 3.00 and float(book_rating) < 4.00:
                            st.write(":star2: :star2: :star2:")
                        elif float(book_rating) > 2.00 and float(book_rating) < 3.00:
                            st.write(":star2: :star2:")
                        elif float(book_rating) > 1.00 and float(book_rating) < 2.00:
                            st.write(":star2:")
                    with g4:
                        st.subheader("🌐 :blue[Search on Google] ")
                        url = 'https://www.google.com/search?'
                        bokk = user_input + ' Book'
                        params = {'q': bokk}
                        url1 = url + urllib.parse.urlencode(params)
                        st.link_button("Click for more info", url1)
         
    except IndexError:
        st.info("Enter full name of the book")
